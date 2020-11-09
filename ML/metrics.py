import torch

# L2 loss for periodic values
class L2AngLoss():
    def __init__(self, period):
        self.period = period

    def __call__(self, x, y):
        loss = x.sub(y)
        return torch.mean(torch.min(loss.pow(2), torch.sub(self.period, loss).pow(2)))

# L1 loss for periodic values
class L1AngLoss():
    def __init__(self, period):
        self.period = period

    def __call__(self, x, y):
        loss = x.sub(y).abs()
        return torch.mean(torch.min(loss, torch.sub(self.period, loss).abs()))

# L1 loss for periodic values, scaled for degrees
class TrueAngLoss():
    def __init__(self, period):
        self.period = period

    def __call__(self, x, y):
        loss = x.sub(y).abs()
        return torch.mean(torch.min(loss, torch.sub(self.period, loss).abs()), 0) / 2 * 360