import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import csv
import config

class EvacuationEnv(gym.Env):
    """ Custom Gym dynamic environment for evacuation simulation """

    def __init__(self):
        """ Initializes environment """
        super(EvacuationEnv, self).__init__()

        # Load the map and set up spaces
        self.load_map()
        self.action_space = spaces.Discrete(5)  # 0: noop, 1: down, 2: up, 3: left, 4: right
        self.observation_space = spaces.Box(0, 1, shape=(8, self.height, self.width), dtype=np.uint8)

        # Initialize display if enabled
        if config.DISPLAY:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * config.CELL_SIZE, self.height * config.CELL_SIZE))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Evacuation Environment")

        self.reset()

    def load_map(self):
        """ Loads map from a CSV file """
        with open(config.MAP_FILE, newline='') as csvfile:
            reader = list(csv.reader(csvfile))
            self.grid = np.array([[int(cell) for cell in row] for row in reader])

        # Update map dimensions
        config.GRID_SIZE = self.grid.shape
        self.height, self.width = self.grid.shape

        # Extract exit locations
        self.exits = [(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y, x] == config.EXIT]

    def one_hot_obs(self):
        """ Generates a one-hot encoded observation of the grid and distance channels """
        # Create one-hot encoding for grid values
        one_hot = np.zeros((6, self.height, self.width), dtype=np.uint8)
        for i in range(6):
            one_hot[i] = (self.grid == i).astype(np.uint8)

        # Initialize distance channels
        distances = np.zeros((2, self.height, self.width), dtype=np.float32)

        # Calculate distances to nearest human and exit
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] != config.WALL:
                    # Distance to nearest human
                    human_dists = [abs(x-hx) + abs(y-hy) for hx, hy in self.human_locations]
                    distances[0, y, x] = min(human_dists) if human_dists else self.height + self.width

                    # Distance to nearest exit
                    exit_dists = [abs(x-ex) + abs(y-ey) for ex, ey in self.exits]
                    distances[1, y, x] = min(exit_dists) if exit_dists else self.height + self.width

        # Normalize distances
        max_dist = self.height + self.width
        distances = distances / max_dist

        return np.concatenate([one_hot, distances], axis=0)

    def reset(self):
        """ Resets the environment to its initial state with randomized placements """
        super().reset()
        self.load_map()
        self.current_step = 0

        self.robot_locations = {}
        self.previous_positions = {}
        self.robot_human_map = {}
        self.frozen_robots = {}

        self.human_locations = set()
        self.fire_locations = set()

        # Place fire in valid locations
        possible_fire_locations = [
            (x, y) for y in range(self.height) for x in range(self.width)
            if self.grid[y, x] == config.EMPTY and all(abs(x-ex) > 2 or abs(y-ey) > 2 for ex, ey in self.exits)
            and all(abs(x-ex) + abs(y-ey) > 2 for ex, ey in self.exits)
        ]
        if possible_fire_locations:
            fire_x, fire_y = random.choice(possible_fire_locations)
            self.fire_locations.add((fire_x, fire_y))
            self.grid[fire_y, fire_x] = config.FIRE

        # Place humans away from fire
        possible_human_locations = [
            (x, y) for y in range(self.height) for x in range(self.width)
            if self.grid[y, x] == config.EMPTY and all(abs(x-fx) > 2 or abs(y-fy) > 2 for fx, fy in self.fire_locations)
        ]
        self.human_locations = set(random.sample(possible_human_locations, min(config.NUM_HUMANS, len(possible_human_locations))))
        for x, y in self.human_locations:
            self.grid[y, x] = config.HUMAN

        # Place robots at exits
        robot_positions = random.choices(self.exits, k=config.NUM_ROBOTS)
        for robot_id, (x, y) in enumerate(robot_positions):
            self.robot_locations[robot_id] = (x, y)
            self.grid[y, x] = config.ROBOT
            self.robot_human_map[robot_id] = 0
            self.frozen_robots[robot_id] = 0

        self.empty_exits = {i: 0 for i in range(config.NUM_ROBOTS)}
        return self.one_hot_obs(), {}

    def step(self, actions):
        """ Executes a step in the environment based on the given actions """
        self.current_step += 1
        agent_rewards = {agent_id: 0 for agent_id in self.robot_locations.keys()}
        terminated = False

        # Store current positions before movement
        self.previous_positions = self.robot_locations.copy()

        # Handle robots on exit tiles
        robots_to_remove = []
        for robot_id, (x, y) in self.robot_locations.items():
            if self.grid[y, x] == config.EXIT:
                # Add early exit penalty if needed
                if self.current_step < config.MIN_STEPS_BEFORE_EXIT:
                    agent_rewards[robot_id] += config.EARLY_EXIT_PENALTY

                if self.robot_human_map[robot_id] > 0:
                    # Successful rescue - reset counter
                    agent_rewards[robot_id] += self.robot_human_map[robot_id] * config.HUMAN_RESCUE_REWARD
                    self.empty_exits[robot_id] = 0
                else:
                    # Empty exit - increase counter and apply multiplied penalty
                    self.empty_exits[robot_id] += 1
                    empty_exit_multiplier = min(self.empty_exits[robot_id], 5)  # Cap at 5x
                    agent_rewards[robot_id] += (config.EMPTY_EXIT_PENALTY * empty_exit_multiplier + config.CONSECUTIVE_EMPTY_EXIT_PENALTY)
                robots_to_remove.append(robot_id)

        # Remove robots that have exited
        for robot_id in robots_to_remove:
            self.robot_locations.pop(robot_id, None)
            self.robot_human_map.pop(robot_id, None)
            self.frozen_robots.pop(robot_id, None)

        # Spread fire to neighboring cells
        new_fires = set()
        for fx, fy in self.fire_locations:
            for dx, dy in config.MOVES[1:]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] not in [config.FIRE, config.WALL, config.EXIT]:
                    if np.random.rand() < config.P_FIRE:
                        new_fires.add((nx, ny))

        for nx, ny in new_fires:
            self.grid[ny, nx] = config.FIRE
        self.fire_locations.update(new_fires)

        # Handle humans caught in fire
        dead_humans = set()
        for x, y in self.human_locations:
            if self.grid[y, x] == config.FIRE:
                dead_humans.add((x, y))
        self.human_locations -= dead_humans
        if dead_humans:
            for agent_id in self.robot_locations:
                agent_rewards[agent_id] += len(dead_humans) * config.HUMAN_FIRE_PENALTY

        # Move robots based on actions
        new_robot_locations = {}
        robots_to_remove = []
        for robot_id, (x, y) in self.robot_locations.items():
            # If robot is frozen reduce countdown
            if self.frozen_robots[robot_id] > 0:
                self.frozen_robots[robot_id] -= 1
                # Robot is in fire
                if self.grid[y, x] == config.FIRE:
                    robots_to_remove.append(robot_id)
                    agent_rewards[robot_id] += config.FIRE_PENALTY
                else:
                    new_robot_locations[robot_id] = (x, y)
                continue

            # Randomly freeze robot 
            if np.random.rand() < config.P_FREEZE:
                self.frozen_robots[robot_id] = config.FREEZE_DURATION - 1 
                new_robot_locations[robot_id] = (x, y)
                continue

            # Determine action and calculate new position
            action = actions[robot_id]
            dx, dy = config.MOVES[action]
            nx, ny = x + dx, y + dy

            # Check if new position is out of bounds
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                new_robot_locations[robot_id] = (x, y)
                continue

            cell = self.grid[ny, nx]

            if cell == config.WALL:
                new_robot_locations[robot_id] = (x, y)
            elif cell == config.FIRE:
                self.grid[y, x] = config.EMPTY
                robots_to_remove.append(robot_id)
                agent_rewards[robot_id] += config.FIRE_PENALTY
            elif cell == config.EXIT:
                self.grid[y, x] = config.EMPTY
                agent_rewards[robot_id] += self.robot_human_map[robot_id] * config.HUMAN_RESCUE_REWARD
                robots_to_remove.append(robot_id)
            else:
                self.grid[y, x] = config.EMPTY
                new_robot_locations[robot_id] = (nx, ny)

                # Check if the robot picks up a human
                if (nx, ny) in self.human_locations:
                    self.human_locations.remove((nx, ny)) 
                    self.robot_human_map[robot_id] += 1 
                    agent_rewards[robot_id] += config.HUMAN_PICKUP_REWARD 

        # Remove robots and update locations
        for robot_id in robots_to_remove:
            self.robot_locations.pop(robot_id, None)
            self.robot_human_map.pop(robot_id, None)
            self.frozen_robots.pop(robot_id, None)

        self.robot_locations.update(new_robot_locations)

        # Ensure fire, humans, robots, and exits are visible on the grid
        for fx, fy in self.fire_locations:
            self.grid[fy, fx] = config.FIRE

        for hx, hy in self.human_locations:
            self.grid[hy, hx] = config.HUMAN

        for robot_id, (rx, ry) in self.robot_locations.items():
            self.grid[ry, rx] = config.ROBOT

        for ex, ey in self.exits:
            self.grid[ey, ex] = config.EXIT

        # Efficiency penalties and distance-based rewards
        for robot_id, (x, y) in self.robot_locations.items():
            # Update grid with new locations
            self.grid[y, x] = config.ROBOT

            # If not carrying humans
            if self.robot_human_map[robot_id] == 0:
                # Reward for moving towards nearest human
                min_human_dist = float('inf')
                for hx, hy in self.human_locations:
                    dist = abs(x - hx) + abs(y - hy)
                    min_human_dist = min(min_human_dist, dist)

                if min_human_dist != float('inf'):
                    agent_rewards[robot_id] += config.HUMAN_DISTANCE_PENALTY * min_human_dist

                continue

            # If carrying humans - reward for moving towards nearest exit
            min_exit_dist = float('inf')
            for ex, ey in self.exits:
                dist = abs(x - ex) + abs(y - ey)
                min_exit_dist = min(min_exit_dist, dist)
            agent_rewards[robot_id] += config.EXIT_DISTANCE_PENALTY * min_exit_dist

        # Add movement rewards
        for robot_id, (x, y) in self.robot_locations.items():
            if robot_id in self.previous_positions:
                prev_x, prev_y = self.previous_positions[robot_id]
                distance_moved = abs(x - prev_x) + abs(y - prev_y) # Manhattan distance
                if distance_moved > 0:
                    agent_rewards[robot_id] += config.MOVEMENT_REWARD * distance_moved
                else:
                    agent_rewards[robot_id] += config.STAYING_STILL_PENALTY

        # Terminate if no robots left
        if len(self.robot_locations) == 0:
            terminated = True
            # Add final penalty for any remaining humans
            if self.human_locations:
                for agent_id in agent_rewards.keys():
                    agent_rewards[agent_id] += len(self.human_locations) * config.HUMAN_FIRE_PENALTY

        # Terminate if max steps reached
        if self.current_step >= config.MAX_ENV_STEPS:
            terminated = True

        # Ensure exits remain visible
        for (x, y) in self.exits:
            self.grid[y, x] = config.EXIT

        return self.one_hot_obs(), agent_rewards, terminated, False, {"positions": self.robot_locations}

    def render(self):
        """ Displays environment """
        if not config.DISPLAY:
            return

        self.screen.fill((255, 255, 255))

        for x in range(config.GRID_SIZE[0]):
            for y in range(config.GRID_SIZE[1]):
                cell_type = self.grid[y, x]

                # Check for frozen robot
                if cell_type == config.ROBOT:
                    for robot_id, (rx, ry) in self.robot_locations.items():
                        if (rx, ry) == (x, y) and self.frozen_robots[robot_id] > 0:
                            cell_type = 6 

                pygame.draw.rect(self.screen, config.COLOR_MAP[cell_type],
                            (x * config.CELL_SIZE, y * config.CELL_SIZE, config.CELL_SIZE, config.CELL_SIZE))
                pygame.draw.rect(self.screen, (0, 0, 0),
                            (x * config.CELL_SIZE, y * config.CELL_SIZE, config.CELL_SIZE, config.CELL_SIZE), 1)

        pygame.display.flip()
        pygame.event.get()
        pygame.time.delay(75)

    def close(self):
        """ Closes display """
        if config.DISPLAY:
            pygame.quit()

    def get_action_mask(self, agent_id):
        """ Returns a boolean mask of valid actions for the given agent """
        if agent_id not in self.robot_locations:
            return np.zeros(5, dtype=bool)

        # Initially all actions are valid
        x, y = self.robot_locations[agent_id]
        mask = np.ones(5, dtype=bool)

        # Check for invalid moves
        for action, (dx, dy) in enumerate(config.MOVES):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height) or self.grid[ny, nx] == config.WALL:
                mask[action] = False

        return mask
