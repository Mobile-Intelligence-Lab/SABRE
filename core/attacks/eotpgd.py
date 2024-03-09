import torch
import torch.nn as nn

from .attack import Attack


class EoTPGDAttack(Attack):
    r"""
    EoT-PGD Attack
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        iters (int): number of steps. (Default: 40)
        eot_steps (int) : number of models to estimate the mean gradient. (Default: 10)
        bounds (tuple) : minimum and maximum values of features. (Default: (0, 1))
    """

    def __init__(self, model, eps=0.3, alpha=2 / 255, iters=40, eot_steps=10, random_start=True, bounds=(0, 1)):
        super(EoTPGDAttack, self).__init__("EoT-PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.eot_iter = eot_steps
        self.random_start = random_start
        self.bounds = bounds

    def forward(self, inputs, labels):
        inputs = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        ori_images = inputs.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            inputs = inputs + torch.empty_like(inputs).uniform_(-self.eps, self.eps)
            inputs = torch.clamp(inputs, min=self.bounds[0], max=self.bounds[1]).detach()

        for _ in range(self.iters):
            grad = torch.zeros_like(inputs)
            inputs.requires_grad = True

            for j in range(self.eot_iter):
                outputs = self.model(inputs)
                cost = loss(outputs, labels).to(self.device)

                # Update adversarial images
                grad += torch.autograd.grad(cost, inputs,
                                            retain_graph=False,
                                            create_graph=True)[0]

            grad /= float(self.eot_iter)

            adv_images = inputs.detach() + self.alpha * grad.sign()
            adv_images.data = torch.min(adv_images, ori_images + self.eps)
            adv_images.data = torch.max(adv_images, ori_images - self.eps)
            inputs.data = torch.clamp(adv_images, min=0, max=1)

        adv_images = inputs

        return adv_images


class EoTPGD:
    def __init__(self, eps, alpha, steps, eot_steps=10, random_start=True, bounds=(0, 1)):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.eot_steps = eot_steps
        self.random_start = random_start
        self.bounds = bounds
        self.attack = None

    def set_model(self, model):
        self.attack = EoTPGDAttack(model=model, eps=self.eps, alpha=self.alpha, iters=self.steps,
                                   eot_steps=self.eot_steps, random_start=self.random_start, bounds=self.bounds)

    def __call__(self, x, y):
        return self.attack(x, y)
