from matplotlib import pyplot as plt
import numpy as np
from gymnasium import make

import networks.continuous_ppo as ppo


class InvertedDoublePendulum:
    """
    The Implementation for Inverted Double Pendulum control task.
    """

    def __init__(self) -> None:
        # define hyper-params
        self.hidden_cells = 128
        self.env = make("InvertedDoublePendulum-v4", render_mode="human")
        num_actions = self.env.action_space.shape[0]
        policy_lr = 0.005
        value_lr = 0.005
        lambda_ = 0.95
        epochs = 100
        epsilon = 0.2
        gamma = 0.9
        self.agent = ppo.Model(hidden_cells=self.hidden_cells,
                               actions=num_actions,
                               policy_lr=policy_lr,
                               value_lr=value_lr,
                               lambda_=lambda_,
                               epochs=epochs,
                               epsilon=epsilon,
                               gamma=gamma)
        return_list = []
        for i in range(5_000):
            state = self.env.reset()[0]  # 环境重置
            done = False  # 任务完成的标记
            episode_return = 0  # 累计每回合的reward

            # 构造数据集，保存每个回合的状态数据
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }

            while not done:
                action = self.agent.next_action(state)  # 动作选择
                next_state, reward, done, _, _ = self.env.step(action)  # 环境更新
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                # 更新状态
                state = next_state
                # 累计回合奖励
                episode_return += reward

            # 保存每个回合的return
            return_list.append(episode_return)
            # 模型训练
            self.agent.update(transition_dict)

            # 打印回合信息
            print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

        plt.plot(return_list)
        plt.title('return')
        plt.show()


InvertedDoublePendulum()
