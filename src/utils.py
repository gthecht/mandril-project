#!/usr/bin/env python

import gridworld as World
import maxent as Maxent
import plot as Plot
import trajectory as Trajectory
import solver as Solver
import optimizer as Optimizer

import numpy as np
import matplotlib.pyplot as plt

def setup_mdp(size=5, p_slip=0, location=None):
    """
    Set-up our MDP/GridWorld
    """
    if location is None: location = np.random.randint(0, size**2)
    # create our world
    world = World.IcyGridWorld(size=size, p_slip=p_slip)

    # set up the reward function
    reward = np.zeros(world.n_states)
    # reward = -0.01 * np.ones(world.n_states)
    reward[location] = 1

    # set up terminal states
    terminal = [location]

    return world, reward, terminal


def get_start(world, terminal):
    """
    set up initial probabilities for trajectory generation
    """
    initial = np.zeros(world.n_states)
    # initial[0] = 1.0
    initial = (1 / (world.n_states - 1)) * np.ones(world.n_states)
    initial[terminal[0]] = 0
    start = np.random.choice(1, p=initial)
    return start

def generate_trajectories(
    world,
    reward,
    terminal,
    n_trajectories = 200,
    discount = 0.7,
    weighting = lambda x: x**5
):
    """
    Generate some "expert" trajectories.
    """

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    # initial[0] = 1.0
    initial = (1 / (world.n_states - 1)) * np.ones(world.n_states)
    initial[terminal[0]] = 0

    # generate trajectories
    value = Solver.value_iteration(world.p_transition, reward, discount)
    policy = Solver.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = Trajectory.stochastic_policy_adapter(policy)
    tjs = list(Trajectory.generate_trajectories(n_trajectories, world, policy_exec, initial, terminal))

    return tjs, policy


def maxent(world, terminal, trajectories, theta=None):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = World.state_features(world)

    # choose our parameter initialization strategy:
    # initialize parameters with constant - this won't be used if theta is defined
    # init = Optimizer.Constant(1.0)
    init = Optimizer.Uniform(0, 1)

    # choose our optimization strategy:
    # we select exponentiated gradient descent with linear learning-rate decay
    optim = Optimizer.ExpSga(lr=Optimizer.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    theta, reward = Maxent.irl(world.p_transition, features, terminal, trajectories, optim, init, theta)

    return theta, reward


def maxent_causal(world, terminal, trajectories, discount=0.7, theta=None):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = World.state_features(world)

    # choose our parameter initialization strategy:
    # initialize parameters with constant - this won't be used if theta is defined
    init = Optimizer.Uniform(0, 1)
    # init = Optimizer.Constant(1.0)

    # choose our optimization strategy:
    # we select exponentiated gradient descent with linear learning-rate decay
    optim = Optimizer.ExpSga(lr=Optimizer.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    theta, reward = Maxent.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount, theta)

    # normalize reward such that the maximum is 1:
    reward = reward / reward.max()

    return theta, reward

def plot_rewards(world, reward, expert_policy, trajectories, reward_maxent, reward_untrained):
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }
    fig = plt.figure(figsize=(14,14))
    # show our original reward
    # ax = plt.figure(num='Original Reward').add_subplot(111)
    ax = fig.add_subplot(221)
    Plot.plot_state_values(ax, world, reward, **style)
    plt.draw()
    plt.title("Original Reward")
    # show our expert policies
    # ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    ax = fig.add_subplot(222)
    Plot.plot_stochastic_policy(ax, world, expert_policy, **style)

    for t in trajectories:
        Plot.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

    plt.draw()
    plt.title("Expert Trajectories and Policy")

    # show the computed reward
    # ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
    ax = fig.add_subplot(223)
    Plot.plot_state_values(ax, world, reward_maxent, **style)
    plt.draw()
    plt.title("MaxEnt Reward")

    # show the computed reward
    # ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
    ax = fig.add_subplot(224)
    Plot.plot_state_values(ax, world, reward_untrained, **style)
    plt.draw()
    plt.title("MaxEnt Untrained Reward")

    plt.show()
