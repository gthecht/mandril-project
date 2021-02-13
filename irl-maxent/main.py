#!/usr/bin/env python

import gridworld as W
import maxent as M
import plot as P
import trajectory as T
import solver as S
import optimizer as O

import numpy as np
import matplotlib.pyplot as plt
import time

# CONSTANTS
P_SLIP = 0
SIZE = 5
LOCATION = np.random.randint(1, SIZE**2)

print(LOCATION)

def setup_mdp():
    """
    Set-up our MDP/GridWorld
    """
    # create our world
    world = W.IcyGridWorld(size=SIZE, p_slip=P_SLIP)

    # set up the reward function
    reward = np.zeros(world.n_states)
    # reward[-1] = 1.0
    reward[LOCATION] = 1

    # set up terminal states
    # terminal = [SIZE**2 - 1]
    terminal = [LOCATION]

    return world, reward, terminal


def generate_trajectories(world, reward, terminal):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    n_trajectories = 200
    discount = 0.7
    weighting = lambda x: x**5

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    # initial[0] = 1.0
    initial = (1 / (world.n_states - 1)) * np.ones(world.n_states)
    initial[LOCATION] = 0

    # generate trajectories
    value = S.value_iteration(world.p_transition, reward, discount)
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, initial, terminal))

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


def main():
    startTime = time.time()
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    # set-up mdp
    world, reward, terminal = setup_mdp()

    fig = plt.figure()
    # show our original reward
    # ax = plt.figure(num='Original Reward').add_subplot(111)
    ax = fig.add_subplot(221)
    P.plot_state_values(ax, world, reward, **style)
    plt.draw()
    plt.title("Original Reward")

    # generate "expert" trajectories
    trajectories, expert_policy = generate_trajectories(world, reward, terminal)

    # show our expert policies
    # ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    ax = fig.add_subplot(222)
    P.plot_stochastic_policy(ax, world, expert_policy, **style)

    for t in trajectories:
        P.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

    plt.draw()
    plt.title("Expert Trajectories and Policy")

    # maximum entropy reinforcement learning (non-causal)
    theta_maxent, reward_maxent = maxent(world, terminal, trajectories)

    print("Theta maxEnt: \n{0}".format(theta_maxent))
    # show the computed reward
    # ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
    ax = fig.add_subplot(223)
    P.plot_state_values(ax, world, reward_maxent, **style)
    plt.draw()
    plt.title("MaxEnt Reward")

    # maximum casal entropy reinforcement learning (non-causal)
    theta_maxcausal, reward_maxcausal = maxent_causal(world, terminal, trajectories)

    print("Theta maxEnt - causal: \n{0}".format(theta_maxcausal))
    # show the computed reward
    # ax = plt.figure(num='MaxEnt Reward (Causal)').add_subplot(111)
    ax = fig.add_subplot(224)
    P.plot_state_values(ax, world, reward_maxcausal, **style)
    plt.draw()
    plt.title("MaxEnt Reward (Causal)")

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    plt.show()


if __name__ == '__main__':
    main()
