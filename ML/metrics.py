import torch

class TrueAngLoss():
    def __init__(self, period):
        self.period = period

    def __call__(self, x, y):
        loss = x.sub(y).abs()
        return torch.mean(torch.min(loss, torch.sub(self.period, loss).abs()), 0) / 2 * 360

class L1AngLoss():
    def __init__(self, period):
        self.period = period

    def __call__(self, x, y):
        loss = x.sub(y).abs()
        return torch.mean(torch.min(loss, torch.sub(self.period, loss).abs()))

class L2AngLoss():
    def __init__(self, period):
        self.period = period

    def __call__(self, x, y):
        loss = x.sub(y)
        return torch.mean(torch.min(loss.pow(2), torch.sub(self.period, loss).pow(2)))