# marl-evacuation-project

Aekus Singh Trehan

evacuation_project/
│
│── baselines/
│   │── astar_baseline.py       # Rule-based A* agent
│   │── greedy_baseline.py      # Greedy heuristic agent
│   └── random_baseline.py      # Random-action agent
│
│── environment/
│   │── maps/                   # Stores predefined map layouts
│   │   │── map1.csv
│   │   │── map2.csv
│   │   └── map3.csv
│   └── evacuation_env.py       # Gym environment logic
│
│── mappo_core/
│   │── actor_critic.py         # Actor-Critic model for MAPPO
│   └── mappo_trainer.py        # Trainer logic for MAPPO
│
│── results/                    # Results are from utils/visualization.py
│   │── baseline/               # Stores results for baselines
│   └── mappo/                  # Stores results for MAPPO 
│
│── training/
│   │── eval_baselines.py       # Evaluation script for all baseline agents
│   │── eval_mappo.py           # Evaluation script for MAPPO agents
│   └── train_mappo.py          # Training script for MARL agents
│
│── utils/
│   │── rollout_buffer.py       # GAE + rollout buffer utils
│   └── visualization.py        # Helper functions for rendering results
│
│── .gitignore                  # For Github repo
│── config.py                   # Stores hyperparameters, environment settings
│── main.py                     # Entry point for running the simulation
└── ReadMe.md                   # Documentation


0   Empty space
1   Wall (obstacle)
2   Exit
3   Human (randomized placement)
4   Fire (randomized placement with randomized spreading)
5   Robot

Robots will spawn at a random exits instead of on their own square
