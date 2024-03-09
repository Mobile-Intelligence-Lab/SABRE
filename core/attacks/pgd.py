from advertorch.attacks import LinfPGDAttack


class PGD:
    def __init__(self, eps, alpha, steps, bounds=(0, 1)):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.bounds = bounds
        self.attack = None

    def set_model(self, model):
        self.attack = LinfPGDAttack(model, nb_iter=self.steps, rand_init=True, eps=self.eps, eps_iter=self.alpha,
                                    clip_min=self.bounds[0], clip_max=self.bounds[1])

    def __call__(self, x, y):
        return self.attack(x, y)
