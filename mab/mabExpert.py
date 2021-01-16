import torch

class MabExpert:
    def __init__(self, env, type="perfect", values={}):
        self.env = env
        self.type = type
        self.values = values
        self.experts = {
            "perfect": self.perfect,
            "rand_from_k_best": self.rand_from_k_best,
            "random_rand_from_k": self.random_rand_from_k,
        }

    def get_actions(self, observations_tensor, envs_vec):
        self._means = torch.tensor(envs_vec.means())
        return self.experts[self.type]()

    def perfect(self):
        return self._means.argmax(axis=1)

    def rand_from_k_best(self):
        arms, sorted_arm_inds = self._means.sort(descending=True)
        N = arms.shape[0]
        rand_ind = torch.randint(0, self.values["k"], (N,1))
        return sorted_arm_inds[torch.arange(0,N)[:,None],rand_ind].squeeze()

    def random_rand_from_k(self):
        self.values["k"] = torch.randint(0,values["max_k"])
        return self.rand_from_k_best()
