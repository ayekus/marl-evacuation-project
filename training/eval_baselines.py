import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import pygame
from environment.evacuation_env import EvacuationEnv
from baselines.astar_baseline import AStarBaselineAgent
from baselines.greedy_baseline import GreedyBaselineAgent
from baselines.random_baseline import RandomBaselineAgent
from utils.visualization import plot_metrics_baseline
import config

def evaluate_baseline(env, agent, baseline_name):
    total_rewards = []
    completion_times = []

    for episode in range(config.NUM_EVAL_EPISODES):
        done = False
        total_reward = 0
        steps = 0

        while not done:
            actions = agent.act()

            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            total_reward += sum(reward.values())
            steps += 1

            if config.DISPLAY:
                env.render()
                time.sleep(0.1)

            if config.DISPLAY:
                pygame.event.pump()

        total_rewards.append(total_reward)
        completion_times.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

        if config.DISPLAY:
            pygame.event.pump()
            time.sleep(0.1)

    print(f"\n{baseline_name} Evaluation Results:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Completion Time: {np.mean(completion_times):.2f} steps\n")

    # Plot and save metrics
    plot_metrics_baseline(total_rewards, completion_times, baseline_name)

    env.close()
    if config.DISPLAY:
        pygame.quit()

if __name__ == "__main__":
    print("\nStarting A* Baseline Evaluation...")
    env = EvacuationEnv()
    astar_agent = AStarBaselineAgent(env)
    evaluate_baseline(env, astar_agent, "AStarBaseline")
    env.close()

    print("\nStarting Greedy Baseline Evaluation...")
    env = EvacuationEnv()
    greedy_agent = GreedyBaselineAgent(env)
    evaluate_baseline(env, greedy_agent, "GreedyBaseline")
    env.close()

    print("\nStarting Random Baseline Evaluation...")
    env = EvacuationEnv()
    random_agent = RandomBaselineAgent(env)
    evaluate_baseline(env, random_agent, "RandomBaseline")
    env.close()
