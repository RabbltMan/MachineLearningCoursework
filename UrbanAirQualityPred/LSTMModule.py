from torch import Tensor, zeros
from torch.nn import Module, LSTM, Linear
from typing import Tuple


class LSTMModule(Module):

    def __init__(self, inputSize: int, hiddenSize: int,
                 outputSize: int) -> None:
        super().__init__()
        self.hiddenCell: Tuple[Tensor, Tensor] = zeros()
        self.lstm = LSTM(inputSize, hiddenSize)
        self.linear = Linear(hiddenSize, outputSize)

    def forward(self, X) -> Tensor:
        X, self.hiddenCell = self.lstm(X, self.hiddenCell)
        y = self.linear(X)
        return y
