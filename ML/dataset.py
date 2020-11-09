import pandas as pd
from pathlib import Path
import os, sys

import torch
from torch.utils.data import Dataset, DataLoader

class PMUAngleDataset(Dataset):
    def __init__(self, meta, data, x1tfms=None, x2tfms=None, ytfms=None, idx=None):
        self.meta = Path(meta)
        self.data = Path(data)
        self.x1tfms = x1tfms
        self.x2tfms = x2tfms
        self.ytfms = ytfms
        self.idx = idx

        # Get a list of the image names
        self.data_names = self._get_filenames(data)
        self.xlength = len(pd.read_csv(self.data/self.data_names[0]))

    # Get files in subdirectory
    def _get_filenames(self, root_dir):
        # Initialise a set to add the file names to
        file_set = set()

        # Iterate through the subdirectories
        for dir_, _, files in os.walk(root_dir):
            for file_name in files:
                # Add the relative file path to the set
                rel_dir = os.path.relpath(dir_, root_dir)
                rel_file = os.path.join(rel_dir, file_name)
                file_set.add(rel_file)

        # Convert to a list before returning
        return list(file_set)

    def _df2tensor(self, df):
        x = []

        for col in df.columns:
            x.append(df[col].tolist())
        x = torch.tensor(x)

        return x

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, i):
        pmu = pd.read_csv(self.data / self.data_names[i])
        pmu = self._df2tensor(pmu)

        try:
            name = self.data_names[i]
            scata_path = Path(name[:name.index("_")] + ".csv")
            scata = pd.read_csv(self.meta/scata_path)
            scata = self._df2tensor(scata)

        except Exception as e:
            print(name, "failed to load")
            print(e)

        mag, ang = scata[:len(scata) // 2].squeeze(1), scata[len(scata) // 2:].squeeze(1)

        # Remove
        # pmu = pmu[:-6,:]

        pmu = self.x1tfms(pmu) if self.x1tfms else pmu
        mag = self.x2tfms(mag) if self.x2tfms else mag
        ang = self.ytfms(ang) if self.ytfms else ang

        if self.idx is None:
            return (pmu, mag), ang
        return (pmu, mag), ang[self.idx]

class PMUAnomalyDataset(Dataset):
    def __init__(self, meta, data, tfms=None, sim_t=3, splits=2):
        self.meta = Path(meta)
        self.data = Path(data)
        self.tfms = tfms
        self.sim_t = sim_t
        self.splits = splits

        # Get a list of the image names
        self.data_names = self._get_filenames(data)
        self.xlength = len(pd.read_csv(self.data/self.data_names[0]))

    # Get files in subdirectory
    def _get_filenames(self, root_dir):
        # Initialise a set to add the file names to
        file_set = set()

        # Iterate through the subdirectories
        for dir_, _, files in os.walk(root_dir):
            for file_name in files:
                # Add the relative file path to the set
                rel_dir = os.path.relpath(dir_, root_dir)
                rel_file = os.path.join(rel_dir, file_name)
                file_set.add(rel_file)

        # Convert to a list before returning
        return list(file_set)

    def _df2tensor(self, df):
        x = []
        for col in df.columns:
            x.append(df[col].tolist())
        x = torch.tensor(x)
        return x

    def _time2idx(self, n):
        if n > self.sim_t:
            return self.xlength

        return floor(self.xlength * n / self.sim_t)

    def _make_label(self, df):
        label = [0] * self.xlength

        for _, row in df.iterrows():
            if row["switch_closed"] < self.sim_t:
                close_idx, open_idx = self._time2idx(row["switch_closed"]), self._time2idx(row["switch_opened"])
                label[close_idx : open_idx] = [1] * (open_idx - close_idx)

        return label

    def _label2BCE(self, label, start, end):
        if any(label[start : end]):
            return torch.tensor([0, 1])
        return torch.tensor([1, 0])

    def __len__(self):
        return len(self.data_names) * self.splits

    def __getitem__(self, i, BCE=True):
        idx = i % self.xlength
        shift = i // self.xlength + 1
        window = self.xlength // (self.splits + 1)
        x_start = randrange(0, window)
        x_end = x_start + window

        x = pd.read_csv(self.data/self.data_names[idx])
        x = self._df2tensor(x)

        y_path = Path(self.data_names[idx][:-9] + ".csv")
        y = pd.read_csv(self.meta/y_path)
        y = self._make_label(y)

        if BCE:
            y = self._label2BCE(y, x_start, x_end)
            return x[:, x_start : x_end], y
        return x[:, x_start : x_end], y[x_start : x_end]

    def plot(self, i):
        x, y = self.__getitem__(i, BCE=False)

        fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(7, 1, sharex=True)
        fig.set_figheight(12)
        fig.set_figwidth(15)

        ax0.plot(x[0])
        ax1.plot(x[1])
        ax2.plot(x[2])
        ax3.plot(x[3])
        ax4.plot(x[4])
        ax5.plot(x[5])
        ax6.plot(y)

        ax0.set_title("i_ang")
        ax1.set_title("i_freq")
        ax2.set_title("i_mag")
        ax3.set_title("v_ang")
        ax4.set_title("v_freq")
        ax5.set_title("v_mag")
        ax6.set_title("faults")

        plt.show()