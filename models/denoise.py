import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_features=1, out_features=1, num_layers=17, num_features=64, upsample_count=1):
        super(DenoisingCNN, self).__init__()
        self.out_features = out_features
        layers = [nn.Sequential(nn.Conv2d(in_features, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features, momentum=0.9, eps=1e-04, affine=True),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, self.out_features, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AvgPool2d(2, 2)

        self.upsample_count = upsample_count
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        b, _, m, n = inputs.shape
        c = self.out_features
        inputs = reshape_inputs(inputs)

        for _ in range(self.upsample_count):
            inputs = self.pool(inputs)
            inputs = F.interpolate(inputs, scale_factor=2, mode='nearest')

        outputs = self.layers(inputs)

        if math.sqrt(m * n) > int(math.sqrt(m * n)):
            outputs = outputs.reshape(b, c, -1)[:, :, :m * n].reshape(b, c, m, n)
        return outputs
