# The environment for the gym
import numpy as np

from mazelab import BaseEnv
from mazelab import VonNeumannMotion
from mazelab import Maze
from mazelab.generators import random_shape_maze

from mazelab.solvers import dijkstra_solver

maze_gen = random_shape_maze

import gym
from gym.spaces import Box
from gym.spaces import Discrete

class MazeEnv(BaseEnv):
    def __init__(self, width, height, max_shapes, max_size, allow_overlap, shape):
        super().__init__()

        self.width = width
        self.height = height
        self.max_shapes = max_shapes
        self.max_size = max_size
        self.allow_overlap = allow_overlap
        self.shape = shape
        self.motions = VonNeumannMotion()

        self.reset_task()

        # We need the observation space to be square, currently it was just one long vector.
        shape = (self.width * self.height,)
        # shape = self.maze.size
        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=shape, dtype=np.float32)
        self.action_space = Discrete(len(self.motions))

    # creates a new random maze generated with maze_gen
    def new_task(self):
        return maze_gen(width=self.width, height=self.height, max_shapes=self.max_shapes, max_size=self.max_size, allow_overlap=self.allow_overlap, shape=self.shape)

    # Returns num_tasks new tasks (new mazes)
    def sample_tasks(self, num_tasks):
        self.tasks = [self.new_task() for ind in range(num_tasks)]
        return self.tasks

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +1
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        return self.maze.to_value(), reward, done, {}

    # reset the specific task - with the same start and finish
    def reset(self):
        empty = np.where(self.x == 0)
        while True:
            inds = np.random.choice(len(empty[0]), 2, replace=False)

            self.start_idx = [[ empty[0][inds[0]], empty[1][inds[0]] ]]
            self.goal_idx  = [[ empty[0][inds[1]], empty[1][inds[1]] ]]

            self.maze = Maze(self.x)
            self.maze.objects.agent.positions = self.start_idx
            self.maze.objects.goal.positions = self.goal_idx
            if self.is_solvable(): break
        # self.maze.objects.agent.positions = self.start_idx
        # self.maze.objects.goal.positions = self.goal_idx
        return self.maze.to_value()

    # reset the task with different start and goal:
    def reset_task(self, task=None):
        self.x = self.sample_tasks(1)[0]
        return self.reset()

    def is_solvable(self):
        actions = dijkstra_solver(
            self.maze.to_impassable(),
            self.motions,
            self.start_idx[0],
            self.goal_idx[0]
        )
        return actions != None

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()

