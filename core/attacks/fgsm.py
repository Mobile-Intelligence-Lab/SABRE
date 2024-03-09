from advertorch.attacks import FGSM as FGSMAttack


class FGSM:
    def __init__(self, eps, bounds=(0, 1)):
        self.eps = eps
        self.bounds = bounds
        self.attack = None

    def set_model(self, model):
        self.attack = FGSMAttack(model, eps=self.eps, clip_min=self.bounds[0], clip_max=self.bounds[1])

    def __call__(self, x, y):
        return self.attack(x, y)
