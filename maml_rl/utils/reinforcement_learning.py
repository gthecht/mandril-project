import numpy as np

from maml_rl.utils.torch_utils import weighted_mean, to_numpy
from mandril.maxent import MaxEnt

def value_iteration(transitions, rewards, gamma=0.95, theta=1e-5):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    delta = np.inf
    while delta >= theta:
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        new_values = np.max(q_values, axis=1)
        delta = np.max(np.abs(new_values - values))
        values = new_values

    return values

def value_iteration_finite_horizon(transitions, rewards, horizon=10, gamma=0.95):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    for k in range(horizon):
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        values = np.max(q_values, axis=1)

    return values

def get_returns(episodes):
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])

# Here I can change the loss to use max-ent. Remember that I need dL/dr.
def reinforce_loss(alg, policy, episodes, params=None):
    if (alg == "mandril"):
        return mandril_reinforce_loss(policy, episodes, params=params)
    else:
        return maml_reinforce_loss(policy, episodes, params=params)

def mandril_reinforce_loss(policy, episodes, params=None):
    max_ent = MaxEnt()
    losses = max_ent.calc(policy, episodes, params)
    return losses


def maml_reinforce_loss(policy, episodes, params=None):
    # I assume that pi the trajectory chosen by the policy. We need to change
    # this to the demos
    pi = policy(episodes.observations.view((-1, *episodes.observation_shape)),
                params=params)

    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
    log_probs = log_probs.view(len(episodes), episodes.batch_size)

    losses = -weighted_mean(log_probs * episodes.advantages,
                            lengths=episodes.lengths)
    # I may want to return the dL/dr instead of L, since then we simply
    # differentiate by theta
    return losses.mean()
