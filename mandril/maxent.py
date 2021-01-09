import torch
from maml_rl.utils.torch_utils import weighted_mean

class MaxEnt:
    def calc(self, policy, episodes, params=None):
        mu_t = self.state_visitations_policy(policy, params, episodes)
        mu_d = self.state_visitations_traj(episodes.actions.view(-1), episodes.rewards.view(-1), mu_t.shape[1])
        # loss = E_mu_t - mu_d # this is of shape (n, 4)
        loss = torch.sum(mu_t * mu_d, 1)
        loss = loss.view(len(episodes), episodes.batch_size)
        batch_reward = torch.exp(sum(episodes.rewards))  * torch.ones_like(loss)
        loss = weighted_mean(loss * batch_reward,
                            lengths=episodes.lengths)
        return loss.mean()

    def state_visitations_traj(self, demos, rewards, action_shape):
        vecs = torch.eye(action_shape)
        vecs = torch.cat((vecs, torch.zeros(1,action_shape)), 0)
        demos[rewards == 0] = action_shape
        # turn demos - which are tensors of actions, into tensors of vectors:
        visitations = torch.stack([vecs[int(action)].T for action in demos])
        visitations[rewards == 0] = 0
        return visitations
    
    def state_visitations_policy(self, policy, params, episodes):
        pi = policy(episodes.observations.view((-1, *episodes.observation_shape)),
                params=params)
        return pi.probs
