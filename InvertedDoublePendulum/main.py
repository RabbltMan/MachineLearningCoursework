import torch
from gymnasium import make


class InvertedDoublePendulum:
    """
    The Implementation for Inverted Double Pendulum control task.
    """

    def __init__(self) -> None:
        # define hyper-params
        self.hidden_cells = 128
        self.env = make("InvertedDoublePendulum-v4", render_mode="human")
        num_actions = self.env.action_space.shape[0]



InvertedDoublePendulum()
