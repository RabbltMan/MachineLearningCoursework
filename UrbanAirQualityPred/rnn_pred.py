from typing import Tuple

import torch
from numpy import int16
from pandas import read_csv
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from lstm_module import LSTMModule


class RNNMultiSeqPred:
    """
    Implementation of the multi-seq prediction model with `RNN` architecture.
    """

    def __init__(self, sample_step: int = 1, epochs: int = 100) -> None:
        self.input_sample_step = sample_step
        self.epochs = epochs
        self.data = RNNMultiSeqPred.load_data()
        self.split_dataset()
        self.loss_func = MSELoss()
        self.model = LSTMModule(input_size=10 * self.input_sample_step,
                                hidden_size=100,
                                output_size=10)
        self.optimizer = Adam(self.model.parameters())

    @staticmethod
    def load_data():
        """
        Load data from `Urban Air Quality Pred` dataset file
        """
        return torch.from_numpy(
            read_csv("./UrbanAirQualityPred/Beijingair.csv",
                     header=None).to_numpy(int16))

    def split_dataset(self,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      batch_size: int = 32) -> Tuple[DataLoader]:
        """
        A method which splits dataset into subsets for train, validation and test.
        """
        if not 0 < train_ratio + val_ratio < 1.0:
            raise ValueError(
                f"Expected sum of train and val ratio in (0, 1) but got {train_ratio + val_ratio}"
            )
        data_rows = self.data.shape[0]
        X, y = [], []
        train_size = int(train_ratio * data_rows)
        val_size = int(val_ratio * data_rows)
        for i in range(data_rows - 10):
            X.append(self.data[i:i + 10].flatten())
            y.append(self.data[i + 10])
        X = torch.stack(X)
        y = torch.tensor(y)
        dataset = TensorDataset(X, y)
        indices = list(range(len(dataset)))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(dataset,
                                  batch_size,
                                  shuffle=True,
                                  sampler=train_sampler)
        val_loader = DataLoader(dataset,
                                batch_size,
                                shuffle=True,
                                sampler=val_sampler)
        test_loader = DataLoader(dataset,
                                 batch_size,
                                 shuffle=True,
                                 sampler=test_sampler)
        return train_loader, val_loader, test_loader

    def train(self) -> None:
        """
        A method which 
        """
        for i in range(self.epochs):
            pass
