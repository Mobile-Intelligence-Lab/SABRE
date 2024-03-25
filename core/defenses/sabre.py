import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from advertorch.bpda import BPDAWrapper


class Sabre(nn.Module):
    def __init__(self, eps: float = 8. / 255, wave='coif1', use_rand: bool = True, n_variants: int = 10):
        super().__init__()

        # Initialization of parameters and setting up the base configuration
        self.eps = eps  # Perturbation magnitude
        self.use_rand = use_rand  # Whether to use randomness
        self.n_variants = n_variants  # Number of variants for randomness
        self.wave = wave  # Wavelet type
        self.mode = 'symmetric'  # Padding mode for DWT
        self.max_levels = 1  # Default max level for wavelet transforms

        self.annealing_coefficient = 1e-12
        self.error_coefficient = 1e-3
        self.use_bins = True

        # Placeholders for forward and inverse wavelet transform functions
        self.fwt = None
        self.iwt = None

    def forward(self, x, lambda_r):
        """Defines the forward pass of the module."""
        return self.transform(x, lambda_r)

    def build_transforms(self, x):
        """Initializes wavelet transforms."""
        _, _, h, w = x.shape
        max_levels = int(math.ceil(math.sqrt(math.log(1 + h * w))))
        self.max_levels = max_levels

        # Initialize DWT and IDWT with the calculated max levels and specified configurations
        self.fwt = DWTForward(J=max_levels, wave=self.wave, mode=self.mode).to(x.device)
        self.iwt = DWTInverse(wave=self.wave, mode=self.mode).to(x.device)

    def reshape_inputs(self, x):
        """Reshapes inputs to have dimensions as squares if necessary."""
        b, c, h, w = x.shape
        target_dim = int(math.ceil(math.sqrt(h * w)))
        if h * w < target_dim ** 2:
            padding_size = target_dim ** 2 - h * w
            x_padded = torch.cat([x.view(b, c, -1), torch.zeros(b, c, padding_size, device=x.device)], dim=2)
            x = x_padded.view(b, c, target_dim, target_dim)
        return x

    def transform(self, x, lambda_r):
        """Applies the transformation process including resizing, wavelet transforms, denoising, and normalization."""
        original_shape = x.shape[-2:]
        x = self.reshape_inputs(x)
        x = self.precision_blend(x)

        if self.use_rand:
            x = self.apply_random_noise(x)

        y = self.apply_wavelet_processing(x, lambda_r)
        y = self.normalize_and_reshape_output(y, original_shape)
        return y

    def precision_blend(self, x, detail_ratio=.1):
        if self.use_bins and self.eps > 0:
            precision = -int(math.floor(math.log10(abs(self.eps)))) - 1
            x = detail_ratio * x + (1 - detail_ratio) * torch.round(x, decimals=precision)
        return x

    def apply_random_noise(self, x):
        """Applies random noise to the input tensor."""
        if not self.use_rand or self.n_variants <= 1:
            return x

        b, c, m, n = x.shape
        xs = x.clone().unsqueeze(2).tile(1, 1, self.n_variants, 1, 1)
        noise = (2 * torch.rand_like(xs, device=x.device) - 1) * self.eps  # Generate noise in the range [-eps, eps]
        noise_coefficients = torch.arange(self.n_variants, device=x.device) / (self.n_variants - 1)
        noise_coefficients = noise_coefficients.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).tile(b, c, 1, m, n)
        noise = noise * noise_coefficients
        xs_noisy = xs + noise
        xs_noisy = torch.clamp(xs_noisy, 0, 1)

        stds = xs_noisy.reshape(b, c, self.n_variants, -1).std(dim=3)
        best_variant = torch.argmin(stds, dim=2)
        batch_indices = torch.arange(b)[:, None].expand(-1, c).to(best_variant.device)
        channel_indices = torch.arange(c)[None, :].expand(b, -1).to(best_variant.device)
        x = xs_noisy[batch_indices, channel_indices, best_variant].reshape(b, c, m, n)

        return x

    def estimate_threshold(self, energies, lambda_r):
        """ Heuristic for estimating the denoising threshold based on spectral energies."""
        threshold = (1 / self.max_levels) * torch.exp(-lambda_r * energies.mean(dim=2) /
                                                      (energies.std(dim=2) + self.error_coefficient))
        threshold = torch.clamp(threshold, min=0, max=1)
        return threshold

    def apply_wavelet_processing(self, x, lambda_r):
        """Applies spectral processing and reconstructs the signal."""
        b, c, m, n = x.shape

        self.build_transforms(x)

        # Apply forward wavelet transform
        x_approx, x_details = self.fwt(x)
        bands = [x_approx.unsqueeze(2)] + x_details

        # Calculate and normalize energies
        energies = torch.cat([(band ** 2).sum(dim=(3, 4)) for band in bands], dim=2)
        min_energies = energies.min(dim=2)[0].unsqueeze(2).tile(1, 1, energies.shape[2])
        max_energies = energies.max(dim=2)[0].unsqueeze(2).tile(1, 1, energies.shape[2])
        energies = (energies - min_energies) / (max_energies - min_energies + self.error_coefficient)

        # Estimate the denoising thresholds
        thresholds = self.estimate_threshold(energies, lambda_r)

        # Denoise spectral bands
        for i, band in enumerate(bands):
            clean_band = band.reshape(b, c, -1)
            coeffs = clean_band.abs().sort(dim=2, descending=True).values
            selection_indices = (thresholds * (coeffs.shape[2] - 1)).floor().to(torch.int64).unsqueeze(-1)
            band_thresholds = torch.gather(coeffs, 2, selection_indices).expand_as(clean_band)
            clean_band[clean_band.abs() < band_thresholds] *= self.annealing_coefficient
            bands[i] = clean_band.reshape(band.shape)

        # Reconstruct input
        y_approx, *y_details = bands
        n_detail_bands = len(y_details)
        y_approx = y_approx.squeeze(dim=2)
        outputs = torch.zeros_like(x, device=x.device).unsqueeze(2).tile(1, 1, n_detail_bands * 3 + 1 + 1, 1, 1)
        zeroed_y_approx = y_approx.clone() * self.annealing_coefficient
        zeroed_y_details = [y_details[j].clone() * self.annealing_coefficient for j in range(n_detail_bands)]

        for i in range(n_detail_bands):
            for k in range(3):
                band_y_details = [details.clone() for details in zeroed_y_details]
                band_y_details[i][:, :, k] = y_details[i][:, :, k]
                outputs[:, :, i * 3 + k] = self.iwt((zeroed_y_approx, band_y_details))
                del band_y_details

        outputs[:, :, n_detail_bands * 3] = self.iwt((y_approx, zeroed_y_details))
        outputs[:, :, n_detail_bands * 3 + 1] = self.iwt((y_approx, y_details))
        outputs = outputs[:, :, :, :m, :n].reshape(b, -1, m, n)

        return outputs

    def normalize_and_reshape_output(self, x, original_shape):
        """Normalizes the output tensor and reshapes it to match the original input shape."""
        b, c, m, n = x.shape
        _m, _n = original_shape

        min_ = x.reshape(*x.shape[:2], -1).min(dim=2)[0].unsqueeze(2).unsqueeze(3).tile(1, 1, m, n)
        max_ = x.reshape(*x.shape[:2], -1).max(dim=2)[0].unsqueeze(2).unsqueeze(3).tile(1, 1, m, n)
        x = (x - min_) / (max_ - min_)

        dim = math.sqrt(_m * _n)
        if dim > int(dim):
            x = x.reshape(b, c, -1)[:, :, :_m * _n].reshape(b, c, _m, _n)

        return x


class SabreWrapper(nn.Module):
    def __init__(self, eps: float = 8. / 255, wave='coif4', use_rand: bool = True, n_variants: int = 10,
                 base_model: nn.Module = None):
        super(SabreWrapper, self).__init__()
        model = Sabre(eps=eps, wave=wave, use_rand=use_rand, n_variants=n_variants)
        self.core = model
        self.base_model = base_model
        self.transform = BPDAWrapper(lambda x, lambda_r: model.transform(x, lambda_r).float())

    @property
    def lambda_r(self):
        return self.base_model.lambda_r

    def reconstruct(self, x):
        return self.base_model.reconstruct(x)

    def preprocessing(self, x):
        b, c, m, n = x.shape

        # process channels independently
        x = x.reshape(-1, 1, m, n)
        x = self.transform(x, self.lambda_r)
        x = self.reconstruct(x)

        return x.reshape(b, c, m, n)

    def classify(self, x):
        return self.base_model.classify(x)

    def features_logits(self, x):
        return self.base_model.features_logits(x)

    def forward(self, x):
        return self.classify(self.preprocessing(x))

    def set_base_model(self, base_model):
        self.base_model = base_model
