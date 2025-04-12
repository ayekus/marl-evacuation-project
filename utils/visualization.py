import matplotlib.pyplot as plt
import os

# Note for the training graphs, tensorboard was used and can be viewed by running tensorboard --logdir runs
# More information about tensorboard can be found here: https://www.tensorflow.org/tensorboard/get_started

def plot_metrics_baseline(rewards, times, baseline_name, save_dir="results/baseline/"):
    os.makedirs(save_dir, exist_ok=True)
    
    episodes = list(range(1, len(rewards) + 1))

    # Plot Rewards
    plt.figure()
    plt.plot(episodes, rewards, marker='o', label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{baseline_name}: Total Reward per Episode")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{baseline_name.lower()}_reward_plot.png"))
    plt.close()

    # Plot Completion Time
    plt.figure()
    plt.plot(episodes, times, marker='s', color='orange', label="Completion Time")
    plt.xlabel("Episode")
    plt.ylabel("Steps Taken")
    plt.title(f"{baseline_name}: Completion Time per Episode")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{baseline_name.lower()}_completion_time_plot.png"))
    plt.close()

def plot_metrics_mappo(rewards, times, save_dir="results/mappo/"):
    os.makedirs(save_dir, exist_ok=True)
    
    episodes = list(range(1, len(rewards) + 1))

    # Plot Rewards
    plt.figure()
    plt.plot(episodes, rewards, marker='o', label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("MAPPO: Total Reward per Episode")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "mappo_reward_plot.png"))
    plt.close()

    # Plot Completion Time
    plt.figure()
    plt.plot(episodes, times, marker='s', color='orange', label="Completion Time")
    plt.xlabel("Episode")
    plt.ylabel("Steps Taken")
    plt.title("MAPPO: Completion Time per Episode")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "mappo_completion_time_plot.png"))
    plt.close()
