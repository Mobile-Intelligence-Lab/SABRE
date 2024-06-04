import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels=1, num_layers=17, num_features=64, classifier=None, eps=1, normalized=False):
        super(DenoisingCNN, self).__init__()
        self.in_channels = in_channels

        layers = [nn.Sequential(nn.Conv2d(3 * in_channels, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1))
        if not normalized:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

        self.classifier = classifier
        self.eps = eps
        self.num_steps = 1
        self.alpha = eps / self.num_steps
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

        self.preprocessing = None
        self.normalize_data = None

    def forward(self, inputs, ctx):
        b, c, m, n = inputs.shape

        initial_logits = self.classifier(inputs.clone())
        initial_preds = initial_logits.max(1)[1]

        ctx_fwd = ctx.clone().detach()
        ctx_bwd = ctx.clone().detach()

        for _ in range(self.num_steps):
            ctx_fwd.requires_grad = True
            ctx_bwd.requires_grad = True
            with torch.enable_grad():
                logits = self.classifier(ctx_fwd)
                loss = F.cross_entropy(logits, initial_preds)
                logits_back = self.classifier(ctx_bwd)
                loss_back = -F.cross_entropy(logits_back, initial_preds)

            grads = torch.autograd.grad(loss, ctx_fwd)[0]
            grads_back = torch.autograd.grad(loss_back, ctx_bwd)[0]
            with torch.no_grad():
                ctx_fwd += self.alpha * grads.sign()
                eta = torch.clamp(ctx_fwd - ctx, min=-self.eps, max=self.eps)
                ctx_fwd = torch.clamp(ctx + eta, min=0, max=1).detach()

                ctx_bwd += self.alpha * grads_back.sign()
                eta = torch.clamp(ctx_bwd - ctx, min=-self.eps, max=self.eps)
                ctx_bwd = torch.clamp(ctx + eta, min=0, max=1).detach()

        inputs = reshape_inputs(inputs)

        ctx_fwd = reshape_inputs(ctx_fwd)
        ctx_bwd = reshape_inputs(ctx_bwd)

        if self.normalize_data is not None:
            ctx_fwd = self.normalize_data(ctx_fwd)
            ctx_bwd = self.normalize_data(ctx_bwd)

        outputs = torch.cat((inputs, ctx_fwd, ctx_bwd), dim=1)

        outputs = self.layers(outputs)

        if math.sqrt(m * n) > int(math.sqrt(m * n)):
            outputs = outputs.reshape(b, c, -1)[:, :, :m * n].reshape(b, c, m, n)

        return outputs
