from math import floor
from random import random
from torchvision import datasets, transforms

import torch

# Rescale tensor to some range
class Rescale():
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.diff = self.upper - self.lower

    def __call__(self, x):

        return x.sub(self.lower).div(self.diff).sub(.5).mul(2)

# Rescale PMU sensor data to some ranges
# Accepts 6 ranges
class RescalePMU():
    def __init__(self, lowers, uppers, num_sensors):
        self.lowers = torch.tensor(lowers * num_sensors).unsqueeze(1)
        self.uppers = torch.tensor(uppers * num_sensors).unsqueeze(1)
        self.diffs = self.uppers - self.lowers

    def __call__(self, x):
        return x.sub(self.lowers).div(self.diffs).sub(.5).mul(2)

# Add white noise at some signal to noise ratio
class AWGN():
    def __init__(self, snr):
        self.snr = snr

    def __call__(self, x):
        powers = x.pow(2).mean((1))
        noise_dbs = 10 * torch.log10(powers) - self.snr
        noise_amps = torch.sqrt(10 ** (noise_dbs / 10))
        return x + torch.randn(x.shape) * noise_amps.unsqueeze(1)

# Random crop within a time series
class RandomCropTS():
    def __init__(self, crop):
        self.crop = crop

    def __call__(self, x):
        split = floor(random() * len(x) * self.crop)
        return torch.cat((x[:,split:], x[:,-1:].expand(-1, split)), 1)

# Cutout some point of the timeseries
# Simulates a drop in connection
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

# Cutout a random channel
# Simulates a drop in connection
class CutoutChannel():
    def __init__(self, pr):
        self.pr = pr

    def __call__(self, x):
        for i in range(len(x)):
            if random() < self.pr:
                x[i, :] = 0

        return x