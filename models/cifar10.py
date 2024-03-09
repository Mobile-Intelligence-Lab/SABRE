import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoise import DenoisingCNN


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        x = x.to(self.conv1.weight.device)
        x_layer1 = F.relu(self.bn1(self.conv1(x)))
        x_layer2 = self.layer1(x_layer1)
        x_layer3 = self.layer2(x_layer2)
        x_layer4 = self.layer3(x_layer3)
        x_layer5 = F.avg_pool2d(self.layer4(x_layer4), 4)

        return x_layer1, x_layer2, x_layer3, x_layer4, x_layer5

    def features_logits(self, x: torch.Tensor):
        x = x.to(self.conv1.weight.device)
        features = self.features(x)
        logits = self.linear(features[-1].view(features[-1].size(0), -1))
        return features, logits

    def forward(self, x):
        features, logits = self.features_logits(x)
        return logits


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)


class Cifar10Model(nn.Module):
    def __init__(self):
        super(Cifar10Model, self).__init__()
        classifier = resnet18()

        self.classifier = classifier
        self.features = classifier.features

        feature_count = 32 * 32
        in_features = int(math.ceil(math.sqrt(math.log(1 + feature_count)))) * 3 + 1 + 1
        self.denoise = DenoisingCNN(in_features=in_features, num_layers=5, num_features=16)

        self.lambda_r = torch.nn.Parameter(torch.ones(1))
        self.normalizers = None

    def set_normalizers(self, normalizers=None):
        self.normalizers = normalizers

    def reconstruct(self, x):
        return self.denoise(x)

    def features_logits(self, x):
        if self.normalizers is not None:
            mean, std = self.normalizers
            x = (x - mean) / std

        return self.classifier.features_logits(x)

    def classify(self, x):
        if self.normalizers is not None:
            mean, std = self.normalizers
            x = (x - mean) / std

        return self.classifier(x)

    def forward(self, x):
        return self.classify(self.reconstruct(x))
