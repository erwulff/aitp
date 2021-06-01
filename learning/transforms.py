import torch


class SmoothLabels(object):
    def __init__(self, alpha, n_classes):
        self.alpha = alpha
        self.n_classes = n_classes

    def __call__(self, sample):
        xb, yb = sample
        yb = torch.mul(yb, (1 - self.alpha))
        return (xb, torch.add(yb, self.alpha / self.n_classes))
