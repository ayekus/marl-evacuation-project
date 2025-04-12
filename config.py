###########################################
# ENVIRONMENT SETUP
###########################################
# Entity representations
EMPTY, WALL, EXIT, HUMAN, FIRE, ROBOT = 0, 1, 2, 3, 4, 5

# Action mappings
ACTION_MAP = {
    (0, 0): 0,  # noop
    (0, -1): 1, # down
    (0, 1): 2,  # up
    (-1, 0): 3, # left
    (1, 0): 4   # right
}
MOVES = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]  # noop, down, up, left, right

# Grid configuration
GRID_SIZE = (10, 10)  # Default size, overridden by map
MAP_FILE = "environment/maps/map2.csv"

# Entity counts
NUM_HUMANS = 5
NUM_ROBOTS = 5

###########################################
# HAZARD PARAMETERS
###########################################
# Fire mechanics
P_FIRE = 0.3            # Probability of fire spreading

# Robot freezing mechanics
P_FREEZE = 0.05         # Probability of freezing per step
FREEZE_DURATION = 3     # Number of steps a robot remains frozen

###########################################
# REWARD STRUCTURE
###########################################
# Primary rewards
HUMAN_RESCUE_REWARD = 1000              # Successfully rescuing a human
HUMAN_PICKUP_REWARD = 200               # Picking up a human

# Penalties
FIRE_PENALTY = -100                     # Robot touching fire
HUMAN_FIRE_PENALTY = -200               # Human touching fire
EMPTY_EXIT_PENALTY = -10                # Exiting without human
EFFICIENCY_PENALTY = -0                 # Base penalty per step
EARLY_EXIT_PENALTY = -1000              # Exiting too early
MIN_STEPS_BEFORE_EXIT = 10              # Minimum steps before allowing exit without penalty 
STAYING_STILL_PENALTY = -0.2            # Not moving

# Distance-based rewards
MOVEMENT_REWARD = 0.1
HUMAN_DISTANCE_PENALTY = -0.5           # Being far from humans
EXIT_DISTANCE_PENALTY = -0.03           # Being far from exit when carrying humans
CONSECUTIVE_EMPTY_EXIT_PENALTY = -25    # Multiple empty exits

###########################################
# TRAINING PARAMETERS
###########################################
# Episode constraints
MAX_ENV_STEPS = 40              # Maximum steps per episode

# Exploration parameters
ENTROPY_COEF = 1.0              # Base exploration rate
MIN_POLICY_STD = 1.0            # Minimum policy randomness
ENTROPY_DECAY = 0.995           # Exploration decay rate
MIN_ENTROPY = 0.1               # Minimum exploration rate

# Training hyperparameters
NUM_EPISODES = 1000000          # Total training episodes
LR = 0.0001                     # Learning rate
GAMMA = 0.99                    # Discount factor
LAMBDA = 0.95                   # GAE parameter
NUM_WORKERS = 1                 # Parallel workers

###########################################
# LOGGING AND CHECKPOINTS
###########################################
LOG_INTERVAL = 500              # Episodes between logs
NUM_EVAL_EPISODES = 100         # Evaluation episodes
CHECKPOINT_DIR = "checkpoints"
LOAD_CHECKPOINT = True
CHECKPOINT_FREQUENCY = 1000     # Episodes between checkpoints

###########################################
# VISUALIZATION
###########################################
DISPLAY = False
COLOR_MAP = {
    0: (200, 200, 200),  # Empty space
    1: (0, 0, 0),        # Wall
    2: (255, 255, 0),    # Exit
    3: (0, 255, 0),      # Human
    4: (255, 0, 0),      # Fire
    5: (0, 0, 255),      # Robot
    6: (0, 0, 55)        # Frozen Robot
}
CELL_SIZE = 50  # Size of each cell in pixels
