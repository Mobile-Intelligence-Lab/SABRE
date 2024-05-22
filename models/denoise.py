import math
import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)


def reshape_inputs(x):
    """Reshapes inputs to have dimensions as squares if necessary."""
    b, c, h, w = x.shape
    target_dim = int(math.ceil(math.sqrt(h * w)))
    if h * w < target_dim ** 2:
        padding_size = target_dim ** 2 - h * w
        x_padded = torch.cat([x.view(b, c, -1), torch.zeros(b, c, padding_size, device=x.device)], dim=2)
        x = x_padded.view(b, c, target_dim, target_dim)
    return x


class DenoisingCNN(nn.Module):
    def __init__(self, in_channels=1, out_features=1, num_layers=17, num_features=64):
        super(DenoisingCNN, self).__init__()
        self.in_channels = in_channels
        layers = [nn.Sequential(nn.Conv2d(2 * in_channels, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs, x0):
        b, _, m, n = inputs.shape
        c = self.in_channels

        inputs = reshape_inputs(inputs)
        x0 = reshape_inputs(x0)

        inputs = torch.cat((inputs, x0), dim=1)

        inputs = ((inputs * (v := ((1 << 3) - 1))).round() - ((inputs < 0) & (inputs % 1 != 0)).float()) / v

        outputs = self.layers(inputs)

        if math.sqrt(m * n) > int(math.sqrt(m * n)):
            outputs = outputs.reshape(b, c, -1)[:, :, :m * n].reshape(b, c, m, n)

        return outputs
