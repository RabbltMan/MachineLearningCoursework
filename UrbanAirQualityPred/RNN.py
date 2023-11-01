from numpy import int16
from pandas import read_csv
from torch import from_numpy
from torch.nn import MSELoss
from torch.optim import Adam
from LSTMModule import *


class RNN(object):

    def __init__(self) -> None:
        self.loadData()
        self.model = LSTMModule(10, 100, 10)
        self.lossFunc = MSELoss()
        self.optimizer = Adam(self.model.parameters())

    def loadData(self):
        self.data = from_numpy(
            read_csv("./UrbanAirQualityPred/Beijingair.csv",
                     header=None).to_numpy(int16))
