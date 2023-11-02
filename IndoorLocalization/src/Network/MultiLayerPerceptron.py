from torch import nn


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(7, 12)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(12, 4)

    def forward(self, x):
        x = self.flatten(x).to(self.linear1.weight.dtype)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
