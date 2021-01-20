import numpy as np
import json
import torch

class ExpertTest():
    def __init__(self, path):
        self.path = path
        self.load_test_data()
        self.load_config()

    def load_test_data(self):
        with np.load(self.path + "/results.npz", allow_pickle=True) as data_file:
            self._logs = {
                "tasks": data_file["tasks"],
                "train_returns" : data_file["train_returns"],
                "train_actions" : data_file["train_actions"],
                "valid_returns" : data_file["valid_returns"],
                "valid_actions" : data_file["valid_actions"],
            }
        self.split_valid_actions()

    def split_valid_actions(self):
        self._logs["valid_post_train"] = self._logs["valid_actions"][:,1,:]
        self._logs["valid_pre_train"] = self._logs["valid_actions"][:,0,:]

    @property
    def logs(self):
        return self._logs

    def load_config(self):
        with open(self.path + "/config.json", 'r') as f:
            self._config = json.load(f)
            if 'env-kwargs' not in self._config.keys(): \
                self._config['env-kwargs'] = {}

    @property
    def config(self):
        return self._config

    def get_sorted(self):
        self.train_sorted = torch.zeros(self._logs["train_actions"].shape)
        self.valid_sorted = torch.zeros(self._logs["valid_actions"].shape)
        self.post_train_sorted = torch.zeros(self._logs["valid_post_train"].shape)
        for ind in range(len(self._logs["tasks"])):
            self.train_sorted[ind,:] = self.get_action_sorted(
                self._logs["tasks"][ind],
                self._logs["train_actions"][ind]
            )
            self.valid_sorted[ind,:] = self.get_action_sorted(
                self._logs["tasks"][ind],
                self._logs["valid_actions"][ind]
            )
            self.post_train_sorted[ind,:] = self.get_action_sorted(
                self._logs["tasks"][ind],
                self._logs["valid_post_train"][ind]
            )

            # self.train_sorted[ind,:], self.valid_sorted[ind,:] = \
            #     self.get_action_sorted(self._logs["tasks"][ind], \
            #     self._logs["train_actions"][ind], \
            #     self._logs["valid_actions"][ind])
    
    def get_action_sorted(self, task, actions_tensor):
        means = torch.tensor(task["mean"])
        actins = torch.tensor(actions_tensor).long()
        sorted_arm_inds = means.sort(descending=True)[1].sort()[1]
        sorted = sorted_arm_inds[actins]
        return sorted

    # def get_action_sorted(self, task, train_actions, valid_actions):
    #     means = torch.tensor(task["mean"])
    #     train = torch.tensor(train_actions).long()
    #     valid = torch.tensor(valid_actions).long()
    #     sorted_arm_inds = means.sort(descending=True)[1].sort()[1]
    #     train_sorted = sorted_arm_inds[train]
    #     valid_sorted = sorted_arm_inds[valid]
    #     return train_sorted, valid_sorted
