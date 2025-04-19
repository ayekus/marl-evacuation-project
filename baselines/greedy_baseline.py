import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import heapq
from environment.evacuation_env import EvacuationEnv
import config

class GreedyBaselineAgent:
    def __init__(self, env: EvacuationEnv):
        self.env = env

    def find_shortest_path(self, start, goal):
        """ Finds shortest path between start and goal positions using A* algorithm """
        if not (0 <= start[0] < self.env.grid.shape[0] and 0 <= start[1] < self.env.grid.shape[1]) or \
           not (0 <= goal[0] < self.env.grid.shape[0] and 0 <= goal[1] < self.env.grid.shape[1]):
            return []

        open_set = [(0, start)]
        g_score = {start: 0}
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path if path else []

            for dy, dx in config.MOVES[1:]:
                neighbor = (current[0] + dy, current[1] + dx)
                if 0 <= neighbor[0] < self.env.grid.shape[0] and 0 <= neighbor[1] < self.env.grid.shape[1]:
                    if self.env.grid[neighbor[0], neighbor[1]] != config.WALL:
                        tentative_g = g_score[current] + 1
                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            g_score[neighbor] = tentative_g
                            f_score = tentative_g + abs(goal[0] - neighbor[0]) + abs(goal[1] - neighbor[1])
                            heapq.heappush(open_set, (f_score, neighbor))
                            came_from[neighbor] = current
        return []

    def act(self):
        """ B-lines for closest humans until none left then b-lines towards nearest exit """
        actions = {}
        for r_id, (rx, ry) in self.env.robot_locations.items():
            if r_id not in self.env.robot_human_map:
                continue

            # If there are any humans, go for them
            if len(self.env.human_locations) > 0:
                shortest_path = None
                for h in self.env.human_locations:
                    path = self.find_shortest_path((ry, rx), (h[1], h[0]))
                    if path and (shortest_path is None or len(path) < len(shortest_path)):
                        shortest_path = path

                if shortest_path and len(shortest_path) > 0:
                    next_step = shortest_path[0]
                    if 0 <= next_step[0] < self.env.grid.shape[0] and 0 <= next_step[1] < self.env.grid.shape[1]:
                        dx = next_step[1] - rx 
                        dy = next_step[0] - ry
                        actions[r_id] = config.ACTION_MAP.get((dx, dy), 0)
                        continue

            # If no humans go to closest exit
            shortest_path = None
            for ex, ey in self.env.exits:
                path = self.find_shortest_path((ry, rx), (ey, ex))
                if path and (shortest_path is None or len(path) < len(shortest_path)):
                    shortest_path = path

            if shortest_path and len(shortest_path) > 0:
                next_step = shortest_path[0]
                dx = next_step[1] - rx
                dy = next_step[0] - ry
                actions[r_id] = config.ACTION_MAP.get((dx, dy), 0)
                continue

            # NOOP as fallback
            actions[r_id] = 0

        return actions
