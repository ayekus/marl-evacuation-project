import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.optim import Adam
import numpy as np
import config

class MAPPOTrainer:
    """ Trainer for Multi-Agent PPO (MAPPO) """

    def __init__(self, actor_critic):
        """ Initializes hyperparameters and optimizer """
        self.ac = actor_critic
        self.device = self.ac.device
        self.optimizer = Adam(self.ac.parameters(), lr=config.LR)

    def update(self, all_rollouts):
        """ Updates the policy and value networks using collected rollouts """
        max_batch_size = max(len(rollouts['obs']) for rollouts in all_rollouts.values())

        def pad_and_convert(data_list, dtype):
            padded_list = []
            for data in data_list:
                if len(data) < max_batch_size:
                    pad_size = max_batch_size - len(data)
                    if len(data.shape) > 1:
                        padding = np.zeros((pad_size,) + data.shape[1:])
                    else:
                        padding = np.zeros(pad_size)
                    data = np.concatenate([data, padding], axis=0)
                padded_list.append(torch.tensor(data, dtype=dtype).to(self.device))
            return padded_list

        # Update padding for positions to maintain batch dimension
        obs_list = pad_and_convert([rollouts['obs'] for rollouts in all_rollouts.values()], torch.float32)
        actions = pad_and_convert([rollouts['actions'] for rollouts in all_rollouts.values()], torch.int64)
        returns = pad_and_convert([rollouts['returns'] for rollouts in all_rollouts.values()], torch.float32)
        advantages = pad_and_convert([rollouts['advantages'] for rollouts in all_rollouts.values()], torch.float32)
        old_log_probs = pad_and_convert([rollouts['log_probs'] for rollouts in all_rollouts.values()], torch.float32)
        positions = pad_and_convert([rollouts['positions'] for rollouts in all_rollouts.values()], torch.float32)

        # Normalize advantages
        advantages = torch.cat([torch.tensor(rollouts['advantages']) for rollouts in all_rollouts.values()])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):  # Multiple epochs
            # Forward pass through shared network with actual positions
            action_logits, values = self.ac(obs_list, positions)

            # Calculate losses for all agents
            policy_loss = 0
            value_loss = 0
            entropy_loss = 0

            for i in range(len(obs_list)):
                dist = torch.distributions.Categorical(logits=action_logits[i])
                log_probs = dist.log_prob(actions[i])
                entropy = dist.entropy()

                ratio = torch.exp(log_probs - old_log_probs[i])
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages[i]

                policy_loss += -torch.min(surr1, surr2).mean()
                value_loss += (returns[i] - values).pow(2).mean()
                entropy_loss += -entropy.mean()

            # Normalize losses by number of agents
            n_agents = len(obs_list)
            total_loss = (policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss) / n_agents

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Return loss values for logging
            return {
                'policy_loss': policy_loss.item() / n_agents,
                'value_loss': value_loss.item() / n_agents,
                'entropy_loss': entropy_loss.item() / n_agents,
                'total_loss': total_loss.item()
            }

    def save_model(self, path):
        """ Saves the model and optimizer state to a file """
        torch.save({
            'model_state': self.ac.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path, map_location=None):
        """ Loads the model and optimizer state from a file """
        checkpoint = torch.load(path, map_location=map_location)
        self.ac.load_state_dict(checkpoint['model_state_dict'])
        if isinstance(checkpoint, dict):
            if 'actor_critic_state' in checkpoint:
                self.ac.load_state_dict(checkpoint['actor_critic_state'])
            elif 'model_state' in checkpoint:
                self.ac.load_state_dict(checkpoint['model_state'])
            if 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            self.ac.load_state_dict(checkpoint)
