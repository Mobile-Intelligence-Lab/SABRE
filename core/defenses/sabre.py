import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import math
import torch
from torch import nn

torch.autograd.set_detect_anomaly(True)

from pytorch_wavelets import DWTForward, DWTInverse
from advertorch.bpda import BPDAWrapper


class Sabre(nn.Module):
    def __init__(self, eps: float = 8. / 255, wave='coif4', use_rand: bool = True, n_variants: int = 10):
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

        # Placeholders for forward and inverse wavelet transform functions
        self.fwt = None
        self.iwt = None

    def forward(self, x, lambda_r):
        """Defines the forward pass of the module."""
        return self.transform(x, lambda_r)

    def build_transforms(self, x):
        """Initializes wavelet transforms."""
        if self.fwt is not None:
            return

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
        original_shape = x.shape[-2:]
        x = self.reshape_inputs(x)
        x = self.precision_blend(x)

        x = self.apply_random_noise(x)
        y = self.apply_wavelet_processing(x, lambda_r)
        y = self.restore_shape(y, original_shape)
        return y

    def precision_blend(self, x):
        if self.eps > 0:
            precision = max(min(-int(math.floor(math.log10(abs(.5 * self.eps)))) - 1, 1), 0)
            x = self.diff_round(x, decimals=precision)
        return x

    def diff_round(self, x, decimals=1):
        scale_factor = (10 ** decimals)
        x = x * scale_factor
        diff = (1 + self.error_coefficient) * x - torch.floor(x)
        x = x - diff + torch.where(diff >= 0.5, 1, 0)
        x = x / scale_factor
        return x

    def apply_random_noise(self, x):
        """Applies random noise to the input tensor."""
        if (not self.use_rand) or (self.n_variants <= 1):
            return x

        b, c, m, n = x.shape
        xs = x.clone().unsqueeze(2).tile(1, 1, self.n_variants, 1, 1)
        noise = (2 * torch.rand_like(xs) - 1) * self.eps  # Generate noise in the range [-eps, eps]
        noise_coefficients = torch.arange(self.n_variants, device=x.device) / (self.n_variants - 1)
        noise_coefficients = noise_coefficients.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).tile(b, c, 1, m, n)
        noise = noise * noise_coefficients
        xs_noisy = xs + noise.to(xs.device)
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
        threshold = torch.nan_to_num(threshold, nan=1., posinf=1., neginf=1.)
        if torch.isnan(lambda_r):
            lambda_r.data = torch.ones(1, device=lambda_r.device)
        return threshold

    def apply_wavelet_processing(self, x, lambda_r):
        """Applies spectral processing and reconstructs the signal."""
        b, c, m, n = x.shape

        self.build_transforms(x)
        self.fwt.to(x.device)

        x_approx, x_details = self.fwt(x)
        bands = [x_approx.unsqueeze(2).to(x.device)] + x_details

        energies = torch.cat([(band ** 2).reshape(b, c, -1).sum(dim=2, keepdim=True) for band in bands], dim=2)
        thresholds = self.estimate_threshold(energies, lambda_r)

        y_details = [
            coeff *
            (torch.where(
                ((coeff ** 2) / (coeff ** 2).amax(dim=1, keepdim=True)) > thresholds[i + 1].mean(),
                1.,
                self.annealing_coefficient
            ) + lambda_r - lambda_r.detach()) for i, coeff in enumerate(x_details)
        ]

        outputs = self.iwt((x_approx, y_details))

        return outputs

    def restore_shape(self, x, original_shape):
        """Reshapes the output to match the original input shape."""
        b, c, m, n = x.shape
        _m, _n = original_shape

        dim = math.sqrt(_m * _n)
        if dim > int(dim):
            x = x.reshape(b, c, -1)[:, :, :_m * _n].reshape(b, c, _m, _n)

        return x


class SabreWrapper(nn.Module):
    def __init__(self, eps: float = 8. / 255, wave='coif4', use_rand: bool = True, n_variants: int = 10,
                 base_model: nn.Module = None, use_bpda=True):
        super(SabreWrapper, self).__init__()
        model = Sabre(eps=eps, wave=wave, use_rand=use_rand, n_variants=n_variants)
        self.core = model
        self.base_model = base_model
        self.use_bpda = use_bpda
        self.transform_fn = lambda x, lambda_r: model.transform(x, lambda_r)
        self.transform_bpda = BPDAWrapper(self.transform_fn)

    def transform(self, x, lambda_r):
        if self.use_bpda:
            return self.transform_bpda(x, lambda_r)
        return self.transform_fn(x, lambda_r)

    @property
    def lambda_r(self):
        return self.base_model.lambda_r

    def reconstruct(self, x, y):
        return self.base_model.reconstruct(x, y)

    def preprocessing(self, x):
        b, c, m, n = x.shape

        y = x
        x = self.transform(x, self.lambda_r)
        x = self.reconstruct(x, y)

        return x.reshape(b, c, m, n)

    def classify(self, x):
        return self.base_model(x)

    def features_logits(self, x):
        return self.base_model.features_logits(x)

    def forward(self, x):
        return self.classify(self.preprocessing(x))

    def set_base_model(self, base_model):
        self.base_model = base_model
