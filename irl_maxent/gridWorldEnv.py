from irl-maxent import gridworld as W
from irl-maxent import maxent as M
from irl-maxent import plot as P
from irl-maxent import trajectory as T
from irl-maxent import solver as S
from irl-maxent import optimizer as O

import numpy as np
import gym

from gym import spaces
from gym.utils import seeding

class gridWorldEnv(gym.Env):
    def __init(self, size, p_slip=0, task={}):
        super(gridWorldEnv, self).__init__()
        self.size = size
        self.p_slip = p_slip
        
        # parameters:
        self.lr0 = 0.2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def new_task(self):
        # random location:
        location = np.random.randint(1, self.size**2)

        world = W.IcyGridWorld(size=self.size, p_slip=self.p_slip)
        reward = np.zeros(world.n_states)
        reward[location] = 1
        terminal = [location]
        task = {
            "world": world,
            "reward": reward,
            "terminal": terminal,
            "discount": 0.7,
            "weighting": lambda x: x**5,
        }
        return task

    def sample_tasks(self, num_tasks):
        self.tasks = [self.new_task() for ind in range(num_tasks)]
        return self.tasks

    # reset the task with different start:
    def reset_task(self, task):
        self._task = task
        # set up initial probabilities for trajectory generation
        self.start = (1 / (task["world"].n_states - 1)) * np.ones(task["world"].n_states)
        # The start cannot be the goal
        self.start[task["terminal"]] = 0
        self._task["start"] = self.start

    def reset_task_single(self, task):
        self._task = task
        # set up initial probabilities for trajectory generation
        start_probs = (1 / (task["world"].n_states - 1)) * np.ones(task["world"].n_states)
        # The start cannot be the goal
        start_probs[task["terminal"]] = 0
        # In order that we have the same start allways
        chosen = np.random.choice(
            task["world"].n_states,
            p=self.start
        )
        self.start = np.zeros(task["world"].n_states)
        self.start[chosen] = 1
        self._task["start"] = self.start

    # reset the specific task - with the same start and finish
    def reset(self):
        # note that the start here is random and not identical to the prev
        return self._task

    # def step(self, action):
    #     assert self.action_space.contains(action)
    #     mean = self._means[action]
    #     reward = self.np_random.binomial(1, mean)
    #     observation = np.zeros(1, dtype=np.float32)

    #     return observation, reward, True, {'task': self._task}

    def generate_trajectories(self, task=None, n_trajectories=1):
        """
        Generate some "expert" trajectories.
        task:
        {world, reward, terminal, start, discount, weighting}
        """
        if task is None: task = self._task
        # generate trajectories
        value = S.value_iteration(task["world"].p_transition, task["reward"], task["discount"])
        policy = S.stochastic_policy_from_value(task["world"], value, w=task["weighting"])
        policy_exec = T.stochastic_policy_adapter(policy)
        tjs = list(T.generate_trajectories(n_trajectories, task["world"], policy_exec, task["start"], task["terminal"]))

        return tjs, policy

    def maxent(self, task, trajectories, parameters=None):
        """
        Maximum Entropy Inverse Reinforcement Learning
        """
        # set up features: we use one feature vector per state
        features = W.state_features(task["world"])

        # choose our parameter initialization strategy:
        #   initialize parameters with constant
        init = O.Inherited(parameters)

        # choose our optimization strategy:
        #   we select exponentiated gradient descent with linear learning-rate decay
        optim = O.ExpSga(lr=O.linear_decay(lr0=self.lr0))

        # actually do some inverse reinforcement learning
        theta, reward = M.irl(task["world"].p_transition, features, task["terminal"], trajectories, optim, init)

        return theta, reward


    def maxent_causal(self, task, trajectories, parameters=None):
        """
        Maximum Causal Entropy Inverse Reinforcement Learning
        """
        # set up features: we use one feature vector per state
        features = W.state_features(task["world"])

        # choose our parameter initialization strategy:
        #   initialize parameters with constant
        
        init = O.Inherited(parameters)

        # choose our optimization strategy:
        #   we select exponentiated gradient descent with linear learning-rate decay
        optim = O.ExpSga(lr=O.linear_decay(lr0=self.lr0))

        # actually do some inverse reinforcement learning
        theta, reward = M.irl_causal(task["world"].p_transition, features, task["terminal"], trajectories, optim, init, task["discount"])

        return theta, reward