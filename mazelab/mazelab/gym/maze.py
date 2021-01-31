# Create the maze class, to use within gym

import numpy as np
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color

class Maze(BaseMaze):
    def __init__(self, x):
        self.x = x
        super().__init__()

    @property

    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        agent = Object('agent', 10, color.agent, False, [])
        goal = Object('goal', 100, color.goal, False, [])
        return free, obstacle, agent, goal