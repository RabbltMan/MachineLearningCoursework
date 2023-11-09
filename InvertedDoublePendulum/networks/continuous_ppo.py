import numpy as np
import torch
from torch import cuda, distributions, nn, optim


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
        x = x.to(self.linear_1.weight.dtype)
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
        x = x.to(self.linear_1.weight.dtype)
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
        self.policy_net = PolicyNet(hidden_cells, actions)
        self.value_net = ValueNet(hidden_cells)
        self.policy_optim = optim.Adam(self.policy_net.parameters(), policy_lr)
        self.value_optim = optim.Adam(self.value_net.parameters(), value_lr)
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
        try:
            action_dict = distributions.Normal(mu, std)
        except ValueError:
            action_dict = distributions.Normal(torch.tensor([[0.1]]),
                                               torch.tensor([[0.1]]))
        action = [action_dict.sample().item()]
        return action

    def update(self, transition_dict):
        """
        A method which trains and updates policy and value networks.
        """
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1)
        if cuda.is_available():
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()

        # [b,states] -> [b,1]
        next_states_target = self.value_net(next_states)
        # temporal difference target，which computes state_value of shape [b,1] at current tick
        td_target = rewards + self.gamma * next_states_target * (1 - dones)
        # temporal difference prediction，which computes state_value of shape [b,1]
        td_value = self.value_net(states)
        # temporal difference result [b,1]
        td_delta = td_value - td_target

        # compute GAE advantage
        td_delta = td_delta.cpu().detach().numpy()  # [b,1]
        advantage_list = []
        advantage = 0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lambda_ * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(advantage_list, dtype=torch.float)

        # predict Gaussian Distribution params of current state action
        mu, std = self.policy_net(states)  # [b,1]
        try:
            action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        except ValueError:
            action_dists = distributions.Normal(torch.tensor([[0.1]]).cuda(),
                                               torch.tensor([[0.1]]).cuda())
        # decide action with log probability func from normal distribution
        old_log_prob = action_dists.log_prob(actions)

        # train with current seq
        for _ in range(self.epochs):
            # predict Gaussian Distribution params of current state action
            mu, std = self.policy_net(states)
            try:
                action_dists = torch.distributions.Normal(mu, std)
            except ValueError:
                continue
            # log probability of action Agent takes at tick t in state s under current policy
            log_prob = action_dists.log_prob(actions)
            # ratio for policy update
            ratio = torch.exp(log_prob - old_log_prob).cpu()

            # PPO Clip loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            value_net_loss = torch.mean(-torch.min(surr1, surr2))
            policy_net_loss = torch.mean(
                nn.functional.mse_loss(self.value_net(states),
                                       td_target.detach()))
            # backward propagation
            self.policy_optim.zero_grad()
            self.value_optim.zero_grad()
            policy_net_loss.backward()
            value_net_loss.backward()
            self.policy_optim.step()
            self.value_optim.step()
