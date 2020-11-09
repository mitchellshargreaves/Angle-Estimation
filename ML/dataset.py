import pandas as pd
from pathlib import Path
import os, sys

import torch
from torch.utils.data import Dataset, DataLoader

# PMU and SCADA (mag / angle) dataset constructor
# Assumes PMU data is named <name>_comb.csv
# Assumes corresponding SCADA data is named <name>.csv
# scada: path of SCADA data
# data: path of PMU data
# x1tfms: transforms for PMU data (input1)
# x2tfms: transforms for SCADA magnitudes (input2)
# ytfms: transforms for SCADA angles (labels)
# idx: Optional index to return a specific angle

class PMUAngleDataset(Dataset):
    def __init__(self, scada, data, x1tfms=None, x2tfms=None, ytfms=None, idx=None):
        self.scada = Path(scada)
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

    # Convert a dataframe of continuous values into a tensor
    def _df2tensor(self, df):
        # Treat columns as channels
        x = []
        for col in df.columns:
            x.append(df[col].tolist())

        # Convert to tensor
        x = torch.tensor(x)
        return x

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, i):
        # Read PMU data
        pmu = pd.read_csv(self.data / self.data_names[i])
        pmu = self._df2tensor(pmu)

        # Read SCADA data
        try:
            name = self.data_names[i]
            scada_path = Path(name[:name.index("_")] + ".csv")
            scada = pd.read_csv(self.scada/scada_path)
            scada = self._df2tensor(scada)

        # Catch and return which file failed, to account for mismatches between datasets
        except Exception as e:
            raise Exception(str(name) + " failed to load")

        # Seperate first half of channels (magnitudes), from angles
        mag, ang = scada[:len(scada) // 2].squeeze(1), scada[len(scada) // 2:].squeeze(1)

        # Transformations
        pmu = self.x1tfms(pmu) if self.x1tfms else pmu
        mag = self.x2tfms(mag) if self.x2tfms else mag
        ang = self.ytfms(ang) if self.ytfms else ang

        # Return all angles
        if self.idx is None:
            return (pmu, mag), ang

        # Return the index provided
        return (pmu, mag), ang[self.idx]