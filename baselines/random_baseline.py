import numpy as np
from environment.evacuation_env import EvacuationEnv
import config

class RandomBaselineAgent:
    def __init__(self, env: EvacuationEnv):
        self.env = env

    def get_random_action(self, rx, ry):
        """ Returns a random valid action for the given position """
        valid_actions = []
        # Check each possible move
        for action, (dx, dy) in enumerate(config.MOVES):
            nx, ny = rx + dx, ry + dy
            # Validate bounds and wall collision
            if (0 <= nx < self.env.grid.shape[1] and 
                0 <= ny < self.env.grid.shape[0] and 
                self.env.grid[ny, nx] != config.WALL):
                valid_actions.append(action)

        # NOOP as fallback
        if not valid_actions:
            return 0

        return np.random.choice(valid_actions)

    def act(self):
        actions = {}
        for r_id, (rx, ry) in self.env.robot_locations.items():
            if r_id not in self.env.robot_human_map:
                continue
            actions[r_id] = self.get_random_action(rx, ry)
        return actions
