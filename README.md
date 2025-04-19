# MARL Evacuation Project

## Project Structure

```
evacuation_project/
│
├── baselines/
│   ├── astar_baseline.py       # Rule-based A* agent
│   ├── greedy_baseline.py      # Greedy heuristic agent
│   └── random_baseline.py      # Random-action agent
│
├── environment/
│   ├── maps/                   # Stores predefined map layouts
│   │   ├── map1.csv
│   │   ├── map2.csv
│   │   └── map3.csv
│   └── evacuation_env.py       # Gym environment logic
│
├── mappo_core/
│   ├── actor_critic.py         # Actor-Critic model for MAPPO
│   └── mappo_trainer.py        # Trainer logic for MAPPO
│
├── results/                    # Results are from utils/visualization.py
│   ├── baseline/               # Stores results for baselines
│   └── mappo/                  # Stores results for MAPPO
│
├── training/
│   ├── eval_baselines.py       # Evaluation script for all baseline agents
│   ├── eval_mappo.py           # Evaluation script for MAPPO agents
│   └── train_mappo.py          # Training script for MARL agents
│
├── utils/
│   ├── rollout_buffer.py       # GAE + rollout buffer utils
│   └── visualization.py        # Helper functions for rendering results
│
├── .gitignore                  # For GitHub repo
├── config.py                   # Stores hyperparameters, environment settings
├── main.py                     # Entry point for running the simulation
└── README.md                   # Documentation
```

---

## Map Legend

| Symbol | Description                                |
| ------ | ------------------------------------------ |
| 0      | Empty space                                |
| 1      | Wall (obstacle)                            |
| 2      | Exit                                       |
| 3      | Human (randomized placement)               |
| 4      | Fire (randomized placement with spreading) |
| 5      | Robot                                      |
