import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import time
from environment.evacuation_env import EvacuationEnv
from mappo_core.actor_critic import ActorCritic
from mappo_core.mappo_trainer import MAPPOTrainer
from utils.visualization import plot_metrics_mappo
import config

def evaluate_mappo(model_path):
    """ Evaluates a trained MAPPO model over multiple episodes """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")

    # Initialize environment and metrics
    env = EvacuationEnv()
    total_rewards = []
    completion_times = []

    try:
        # Load the trained model
        shared_ac = ActorCritic(env.one_hot_obs().shape, config.NUM_ROBOTS).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if 'episode' in state_dict:
            state_dict.pop('episode')
        if 'optimizer_state' in state_dict:
            state_dict.pop('optimizer_state')
        if 'best_reward' in state_dict:
            state_dict.pop('best_reward')

        shared_ac.load_state_dict(state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Evaluation loop
    for episode in range(config.NUM_EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            if config.DISPLAY:
                env.render()
                time.sleep(0.1)

            # Prepare observations for all agents
            active_agents = list(env.robot_locations.keys())
            obs_list = []
            pos_list = []
            actions = {}

            for agent_id in range(config.NUM_ROBOTS):
                if agent_id in active_agents:
                    agent_pos = env.robot_locations[agent_id]
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    pos_tensor = torch.FloatTensor(agent_pos).unsqueeze(0).to(device)
                else:
                    obs_tensor = torch.zeros((1,) + env.one_hot_obs().shape).to(device)
                    pos_tensor = torch.zeros(1, 2).to(device)

                obs_list.append(obs_tensor)
                pos_list.append(pos_tensor)

            # Get actions using the policy
            with torch.no_grad():
                action_logits, _ = shared_ac(obs_list, pos_list)

            # Execute actions for active agents
            for idx, agent_id in enumerate(active_agents):
                dist = torch.distributions.Categorical(logits=action_logits[idx])
                action = dist.sample()
                actions[agent_id] = action.item()

            # Step environment
            obs, rewards, terminated, _, info = env.step(actions)
            total_reward += sum(rewards.values())
            done = terminated
            steps += 1

        total_rewards.append(total_reward)
        completion_times.append(steps)

        print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Completion Time: {np.mean(completion_times):.2f} steps")

    # Plot and save metrics
    plot_metrics_mappo(total_rewards, completion_times)

    if config.DISPLAY:
        env.close()

    return np.mean(total_rewards)

if __name__ == "__main__":
    model_path = "checkpoints/latest.pt"  # Try loading latest checkpoint first
    if not os.path.exists(model_path):
        model_path = "checkpoints/best_model.pt"  # Fall back to best model

    reward = evaluate_mappo(model_path)
    if reward is not None:
        print(f"Final Average Reward: {reward:.2f}")
