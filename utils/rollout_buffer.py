import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import config

class RolloutBuffer:
    """ Buffer to store rollouts for PPO training """

    def __init__(self):
        """ Initializes the buffer with discount and lambda values """
        self.gamma = config.GAMMA
        self.lam = config.LAMBDA
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.positions = []

    def store(self, obs, action, reward, done, log_prob, value, position):
        """ Stores a single step of interaction in the buffer """
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.positions.append(position)

    def get(self):
        """ Computes advantages and returns the stored data """
        values = np.array(self.values)
        returns, advantages = compute_gae(
            np.array(self.rewards),
            values,
            np.array(self.dones)
        )

        return {
            'obs': np.array(self.obs),
            'actions': np.array(self.actions),
            'returns': returns,
            'advantages': advantages,
            'log_probs': np.array(self.log_probs),
            'values': values,
            'positions': np.array(self.positions)
        }

    def clear(self):
        """ Clears the buffer """
        self.__init__()

def compute_gae(rewards, values, dones):
    """ Computes Generalized Advantage Estimation (GAE) """
    gamma = config.GAMMA
    lam = config.LAMBDA
    advantages = np.zeros_like(rewards)
    last_gae = 0

    # Append a final value of 0 for terminated episodes
    values = np.append(values, 0)

    for t in reversed(range(len(rewards))):
        if dones[t]:
            last_gae = 0

        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:-1]
    return returns, advantages
