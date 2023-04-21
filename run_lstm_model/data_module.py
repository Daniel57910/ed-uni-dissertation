import pdb

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

LABEL_INDEX = 1
TOTAL_EVENTS_INDEX = 2
BATCHES = 1000000

class ClickstreamDataModule(LightningDataModule):
    def __init__(self, pointer_list, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size
        train_index = int(pointer_list[0].shape[0] * 0.7)
        val_index = int(pointer_list[0].shape[0] * 0.85)
        self.training_data = [p[:train_index] for p in pointer_list]
        self.validation_data = [p[train_index:val_index] for p in pointer_list]
        self.test_data = [p[val_index:] for p in pointer_list]
        self.n_sequences = n_sequences
        self.n_features = n_features

    def train_dataloader(self):
        dataset = ClickstreamDataset(self.training_data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=False)

    def val_dataloader(self):
        dataset = ClickstreamDataset(self.validation_data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=False)
    
    def test_dataloader(self):
        dataset = ClickstreamDataset(self.test_data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=False)
    
class ClickstreamDataset(Dataset):
    def __init__(self, dataset_pointer_list) -> None:
        super().__init__()
        """
        Yield data in batches of BATCHES
        """
        self.pointer_list = dataset_pointer_list
        self.total_events = self.pointer_list[0].shape[0]


    def __getitem__(self, idx):
        result = [np.array(i[idx]) for i in self.pointer_list]
        return np.concatenate(result)

    def __len__(self):
        return self.total_events
