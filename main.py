import os
import torch
from environment.evacuation_env import EvacuationEnv
from mappo_core.actor_critic import ActorCritic
import config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment
    env = EvacuationEnv()
    obs, _ = env.reset()
    obs_shape = env.one_hot_obs().shape
    n_agents = config.NUM_ROBOTS

    # Initialize shared Actor-Critic model
    shared_ac = ActorCritic(obs_shape).to(device)

    # Load model if checkpoint exists
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        shared_ac.load_state_dict(checkpoint['model_state'])
        print(f"Loaded model from {checkpoint_path}")

    # Run simulation loop
    terminated = False
    while not terminated:
        # Prepare observations and positions for all agents
        active_agents = list(env.robot_locations.keys())
        obs_list = []
        pos_list = []

        for agent_id in range(n_agents):
            if agent_id in active_agents:
                agent_pos = env.robot_locations[agent_id]
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                pos_tensor = torch.tensor(agent_pos, dtype=torch.float32, device=device)
            else:
                obs_tensor = torch.zeros((8, env.height, env.width), dtype=torch.float32, device=device)
                pos_tensor = torch.zeros(2, dtype=torch.float32, device=device)

            obs_list.append(obs_tensor)
            pos_list.append(pos_tensor)

        # Get actions using the Actor-Critic model
        with torch.no_grad():
            action_logits, _ = shared_ac(obs_list, pos_list)

        actions = {}
        for idx, agent_id in enumerate(active_agents):
            dist = torch.distributions.Categorical(logits=action_logits[idx])
            actions[agent_id] = dist.sample().item()

        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        env.render()

    print("Simulation completed.")

if __name__ == "__main__":
    main()
