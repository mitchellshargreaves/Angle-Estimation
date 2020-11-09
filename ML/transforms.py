from math import floor
from random import random
from torchvision import datasets, transforms

import torch

# class NormalizeTS():
#     def __init__(self, means, stds):
#         self.means = means.unsqueeze(1)
#         self.stds = stds.unsqueeze(1)
#
#     def __call__(self, x):
#         return x.sub(self.means).div(self.stds)
#
# class Normalize():
#     def __init__(self, means, stds):
#         self.means = means
#         self.stds = stds
#
#     def __call__(self, x):
#         return x.sub(self.means).div(self.stds)

class Normalize0D():
    def __init__(self, means, stds):
        self.tf = transforms.Normalize(means, stds)

    def __call__(self, x):
        return self.tf(x.unsqueeze(1).unsqueeze(2)).squeeze(2).squeeze(1)

class Normalize1D():
    def __init__(self, means, stds):
        self.tf = transforms.Normalize(means, stds)

    def __call__(self, x):
        return self.tf(x.unsqueeze(2)).squeeze(2)

class Rescale():
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.diff = self.upper - self.lower

    def __call__(self, x):

        return x.sub(self.lower).div(self.diff).sub(.5).mul(2)

class RescalePMU():
    def __init__(self, lowers, uppers, num_sensors):
        self.lowers = torch.tensor(lowers * num_sensors).unsqueeze(1)
        self.uppers = torch.tensor(uppers * num_sensors).unsqueeze(1)
        self.diffs = self.uppers - self.lowers

    def __call__(self, x):
        return x.sub(self.lowers).div(self.diffs).sub(.5).mul(2)

class AWGN():
    def __init__(self, snr):
        self.snr = snr

    def __call__(self, x):
        powers = x.pow(2).mean((1))
        noise_dbs = 10 * torch.log10(powers) - self.snr
        noise_amps = torch.sqrt(10 ** (noise_dbs / 10))
        return x + torch.randn(x.shape) * noise_amps.unsqueeze(1)

class RandomCropTS():
    def __init__(self, crop):
        self.crop = crop

    def __call__(self, x):
        split = floor(random() * len(x) * self.crop)
        return torch.cat((x[:,split:], x[:,-1:].expand(-1, split)), 1)

# Detrimental
class RandomAmplify():
    def __init__(self, amp):
        self.amp = amp

    def __call__(self, x):
        for i in range(2, len(x), 3):
            x[i] = x[i] * ((random() - 0.5) * 2 * self.amp + 1)
        return x

class CutoutTime():
    def __init__(self, pr, size):
        self.pr = pr
        self.size = size

    def __call__(self, x):
        if random() < self.pr:
            start_idx = floor(random() * len(x))
            end_idx = min(floor(random() * self.size) + start_idx, len(x))

            x[:, start_idx:end_idx] = 0

        return x

class CutoutChannel():
    def __init__(self, pr):
        self.pr = pr

    def __call__(self, x):
        for i in range(len(x)):
            if random() < self.pr:
                x[i, :] = 0

        return x

class TrueAngLoss():
    def __init__(self, period):
        self.period = period

    def __call__(self, x, y):
        loss = x.sub(y).abs()
        return torch.mean(torch.min(loss, torch.sub(self.period, loss).abs())) / 2 * 360

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