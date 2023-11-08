import numpy as np
import torch
from torch import cuda, distributions, nn, optim

import networks.continuous_ppo as ppo


class PolicyNet(nn.Module):
    """
    ### PPO Policy
    PPO Algorithm handles exploration through a stochastic policy, 
    which outputs parameters of a distribution.
    """

    def __init__(self, hidden_cells: int, actions: int) -> None:
        super().__init__()
        self.linear_1 = nn.LazyLinear(hidden_cells)
        self.relu = nn.Tanh()
        self.linear_2 = nn.LazyLinear(actions)
        self.tanh = nn.Tanh()
        self.linear_3 = nn.LazyLinear(actions)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        The forward propagation of `PPO Policy Network`
        """
        x = self.linear_1(x)
        x = self.relu(x)
        mu = self.linear_2(x)
        mu = 2 * self.tanh(mu)
        std = self.linear_3(x)
        std = self.softplus(std)
        return mu, std


class ValueNet(nn.Module):
    """
    ### PPO Value
    `ValueNet` reads the observations and returns an 
    estimation of the discounted return for the following trajector.
    """

    def __init__(self, hidden_cells: int) -> None:
        super().__init__()
        self.linear_1 = nn.LazyLinear(hidden_cells)
        self.relu = nn.ReLU()
        self.linear_2 = nn.LazyLinear(1)

    def forward(self, x):
        """
        The forward propagation of `PPO Value Network`
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class Model:
    """
    Continuous PPO Model Implementation
    """

    def __init__(self, hidden_cells: int, actions: int, policy_lr: float,
                 value_lr: float, lambda_: float, epochs: int, epsilon: float,
                 gamma: float) -> None:
        self.policy_net = ppo.PolicyNet(hidden_cells, actions)
        self.value_net = ppo.ValueNet(hidden_cells)
        self.policy_optim = optim.Adam(self.policy_net.parameters(), policy_lr)
        self.value_optim = optim.Adam(self.value_net, value_lr)
        self.lambda_ = lambda_
        self.epochs = epochs
        self.epsilon = epsilon
        self.gamma = gamma
        # use cuda to accelerate if available
        if cuda.is_available():
            self.policy_net = self.policy_net.cuda()
            self.value_net = self.value_net.cuda()

    def next_action(self, current_state):
        """
        `current_state` -> `next_action` distribution
        """
        current_state = torch.tensor(current_state[np.newaxis, :])
        if cuda.is_available():
            current_state = current_state.cuda()
        mu, std = self.policy_net(current_state)
        action_dict = distributions.Normal(mu, std)
        action = list(action_dict.sample().item())
        return action

    def update(self, transtition_dict):
        """
        A method which trains and updates policy and value networks.
        """
        pass
