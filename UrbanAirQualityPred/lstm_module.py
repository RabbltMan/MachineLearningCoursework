from typing import Optional

import torch
from torch.nn import Module, LSTM, Linear



class LSTMModule(Module):
    """
    A specific RNN module with `Long Short Term Memory` sturcture.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int) -> None:
        super().__init__()
        self.lstm: LSTM = LSTM(input_size, hidden_size)
        self.linear: Linear = Linear(hidden_size, output_size)
        self.hidden_cell: Optional[torch.Tensor] = None

    def forward(self, x) -> torch.Tensor:
        """
        Forward propagation of the network.
        """
        x, self.hidden_cell = self.lstm(x.view(len(x), 1, -1))
        y = self.linear(x)
        return y
