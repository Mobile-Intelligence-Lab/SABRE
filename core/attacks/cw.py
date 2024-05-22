import torch
from advertorch.attacks import CarliniWagnerL2Attack


class CW:
    def __init__(self, c=1e-4, kappa=0, steps=1000, lr=0.01, eps=1., bounds=(0, 1), num_classes=1):
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.eps = eps
        self.bounds = bounds
        self.attack = None
        self.num_classes = num_classes

    def set_model(self, model):
        self.attack = CarliniWagnerL2Attack(model, num_classes=self.num_classes, confidence=self.kappa,
                                            learning_rate=self.lr, max_iterations=self.steps, initial_const=self.c,
                                            clip_min=self.bounds[0], clip_max=self.bounds[1])

    def __call__(self, x, y):
        x_advs = self.attack(x, y)
        x_advs = torch.max(torch.min(x_advs, x + self.eps), x - self.eps)
        x_advs = torch.clamp(x_advs, min=self.bounds[0], max=self.bounds[1])
        return x_advs
