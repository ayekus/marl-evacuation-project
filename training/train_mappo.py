import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from environment.evacuation_env import EvacuationEnv
from mappo_core.actor_critic import ActorCritic
from mappo_core.mappo_trainer import MAPPOTrainer
from utils.rollout_buffer import RolloutBuffer
import config

def main():
    """ Main training loop for MAPPO """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment, model, and trainer
    env = EvacuationEnv()
    obs, _ = env.reset()
    obs_shape = env.one_hot_obs().shape
    n_agents = config.NUM_ROBOTS

    # Create single shared network for all agents
    shared_ac = ActorCritic(obs_shape, n_agents).to(device)
    trainer = MAPPOTrainer(shared_ac)
    buffers = {agent_id: RolloutBuffer() for agent_id in range(n_agents)}

    writer = SummaryWriter()

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_reward = float('-inf')
    start_episode = 0

    # Load checkpoint if available
    latest_checkpoint = os.path.join(config.CHECKPOINT_DIR, "latest.pt")
    start_episode = 0
    best_reward = float('-inf')
    avg_reward = float('-inf')
    total_episodes = config.NUM_EPISODES

    if config.LOAD_CHECKPOINT and os.path.exists(latest_checkpoint):
        try:
            checkpoint = torch.load(latest_checkpoint, weights_only=True)
            shared_ac.load_state_dict(checkpoint['model_state'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])

            # Validate checkpoint values
            if 'episode' in checkpoint and isinstance(checkpoint['episode'], int):
                start_episode = checkpoint['episode'] + 1
                total_episodes = start_episode + config.NUM_EPISODES
            if 'best_reward' in checkpoint and isinstance(checkpoint['best_reward'], (int, float)):
                best_reward = checkpoint['best_reward']

            print(f"Loaded checkpoint from episode {start_episode-1}")
            print(f"Will train for {config.NUM_EPISODES} more episodes (until episode {total_episodes})")
            print(f"Best reward so far: {best_reward}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training")
            start_episode = 0
            total_episodes = config.NUM_EPISODES

    # Add running statistics
    running_rewards = []
    window_size = 100

    print("\nStarting training...")
    print(f"Training from episode {start_episode} to {total_episodes}")
    print("-" * 50)

    # Training loop
    pbar = tqdm(range(start_episode, total_episodes), desc="Training", ncols=150)
    for episode in pbar:
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        total_agents_active = 0

        while not done:
            # Prepare observations and positions for all agents
            active_agents = list(env.robot_locations.keys())
            total_agents_active += len(active_agents)
            obs_list = []
            pos_list = []

            for agent_id in range(n_agents):
                if agent_id in active_agents:
                    agent_pos = env.robot_locations[agent_id]
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)  # Shape: (8, H, W)
                    pos_tensor = torch.tensor(agent_pos, dtype=torch.float32)
                else:
                    obs_tensor = torch.zeros((8, env.height, env.width), dtype=torch.float32)
                    pos_tensor = torch.zeros(2, dtype=torch.float32)

                obs_list.append(obs_tensor)
                pos_list.append(pos_tensor)

            # Move to device after all tensors are created
            obs_list = [obs.to(device) for obs in obs_list]
            pos_list = [pos.to(device) for pos in pos_list]

            # Get actions and values using shared policy
            with torch.no_grad():
                action_logits, value = shared_ac(obs_list, pos_list)

            actions = {}
            # Only process actions for active agents
            for idx, agent_id in enumerate(active_agents):
                action_mask = env.get_action_mask(agent_id)
                logits = action_logits[idx]

                # print(f"Agent {agent_id} - Action distribution:", logits)
                # print(f"Agent {agent_id} - Action mask:", action_mask)

                # Convert mask to tensor and move to correct device
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device)

                # Apply mask by setting invalid actions to large negative values
                masked_logits = torch.where(
                    mask_tensor,
                    logits,
                    torch.tensor(-1e8, device=device)
                )

                # Sample action using masked logits
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # print(f"Agent {agent_id} - Masked logits:", masked_logits)
                # print(f"Agent {agent_id} - Sampled action:", action.item())

                actions[agent_id] = action.item()
                buffers[agent_id].store(
                    obs_list[idx].squeeze(0).cpu().numpy(),
                    action.item(),
                    0,  # reward placeholder
                    False,  # done placeholder
                    log_prob.cpu().item(),
                    value.cpu().item(),
                    agent_pos  # Store agent position
                )

            next_obs, rewards, terminated, _, info = env.step(actions)
            done = terminated

            # Update rewards in buffers
            for agent_id, reward in rewards.items():
                buffers[agent_id].rewards[-1] = reward

            obs = next_obs
            episode_reward = sum(rewards.values())
            step += 1

        # Calculate running statistics
        running_rewards.append(episode_reward)
        if len(running_rewards) > window_size:
            running_rewards.pop(0)
        avg_reward = sum(running_rewards) / len(running_rewards)

        # Update progress bar with more informative metrics
        pbar.set_postfix({
            'last_reward': f'{episode_reward:.1f}',
            'avg_reward': f'{avg_reward:.1f}',
            'steps': step,
        })

        # Log to tensorboard
        writer.add_scalar("Metrics/EpisodeReward", episode_reward, episode)
        writer.add_scalar("Metrics/AverageReward", avg_reward, episode)
        writer.add_scalar("Metrics/Steps", step, episode)

        # Update policy using centralized training
        all_rollouts = {agent_id: buffer.get() for agent_id, buffer in buffers.items()}
        trainer.update(all_rollouts)

        # Clear buffers
        for buffer in buffers.values():
            buffer.clear()

        # Save checkpoints and log metrics
        checkpoint = {
            'episode': episode,
            'model_state': shared_ac.state_dict(),
            'optimizer_state': trainer.optimizer.state_dict(),
            'best_reward': best_reward
        }
        torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, "latest.pt"))

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))
            pbar.write(f"New best reward: {best_reward:.2f}")

        if episode % config.CHECKPOINT_FREQUENCY == 0:
            torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, f"model_ep_{episode}.pt"))
            pbar.write(f"Checkpoint saved at episode {episode}")

        # Print milestone information
        if episode % 100 == 0:
            print(f"\nEpisode {episode}:")
            print(f"  Average reward (last {window_size}): {avg_reward:.2f}")
            print(f"  Best reward so far: {best_reward:.2f}")
            print(f"  Average steps: {step:.1f}")
            print("-" * 50)

    # Final statistics
    print("\nTraining completed!")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Final average reward: {avg_reward:.2f}")
    writer.close()

if __name__ == "__main__":
    """ Entry point for training script """
    main()
