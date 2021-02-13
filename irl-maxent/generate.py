import gridworld as W
import maxent as M
import plot as P
import trajectory as T
import solver as S
import optimizer as O

import numpy as np

def generate_trajectories(task, n_trajectories):
    """
    Generate some "expert" trajectories.
    task:
     {world, reward, terminal, start, discount, weighting}
    """

    # generate trajectories
    value = S.value_iteration(task["world"].p_transition, task["reward"], task["discount"])
    policy = S.stochastic_policy_from_value(task["world"], value, w=task["weighting"])
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories, task["world"], policy_exec, task["start"], task["terminal"]))

    return tjs, policy


def maxent(world, terminal, trajectories):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = W.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    theta, reward = M.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return theta, reward


def maxent_causal(world, terminal, trajectories, discount=0.7):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = W.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    theta, reward = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return theta, reward