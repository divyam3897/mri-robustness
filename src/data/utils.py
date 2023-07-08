import math
import os
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from joblib import dump, load
from torch.utils import data
from torch.utils.data import (DataLoader, Dataset, WeightedRandomSampler,
                              sampler)
from tqdm.auto import tqdm


class MultiDataset(Dataset):
    def __init__(self, split_csv_file: str, mode: str, dev_mode: bool = False):
        super().__init__()
        # read csv file with filenames
        self.split_csv_file = Path(split_csv_file)
        assert self.split_csv_file.is_file()

        self.metadata = pd.read_csv(self.split_csv_file)
        if dev_mode:
            self.metadata = self.metadata.iloc[:1500]

        assert "data_split" in self.metadata.columns
        assert "location" in self.metadata.columns

        metadata_grouped = self.metadata.groupby("data_split")
        self.metadata_by_mode = {
            e: metadata_grouped.get_group(e) for e in metadata_grouped.groups
        }

        self.mode = mode
        assert mode in self.metadata_by_mode

    def __len__(self):
        assert self.mode in self.metadata_by_mode
        return self.metadata_by_mode[self.mode].shape[0]

    def get_metadata_value(self, index, key):
        assert self.mode in self.metadata_by_mode
        assert key in self.metadata_by_mode[self.mode].iloc[index]
        return self.metadata_by_mode[self.mode].iloc[index][key]

    def __getitem__(self, index):
        raise NotImplementedError


if __name__ == "__main__":
    pass
