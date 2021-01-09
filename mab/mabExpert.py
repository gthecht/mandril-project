import torch

class MabExpert:
    def __init__(self, env):
        self.env = env

    def get_actions(self, observations_tensor, envs_vec):
        self._means = torch.tensor(envs_vec.means())
        return self._means.argmax(axis=1)