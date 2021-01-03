import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init

class CategoricalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Categorical` distribution output. This policy network can be used on tasks 
    with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_categorical_mlp_policy.py
    """
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=(),
                 nonlinearity=F.relu):
        super(CategoricalMLPPolicy, self).__init__(input_size=input_size,
                                                   output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (1,) + hidden_sizes + (output_size,)
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            nn.Conv2d(layer_sizes[i - 1], layer_sizes[i], 3))
        self.add_module('layer{0}'.format(self.num_layers),
                        nn.Linear(
                            layer_sizes[-2] *
                            (input_size[0] - 2 * len(hidden_sizes)) *
                            (input_size[1] - 2 * len(hidden_sizes)),
                            layer_sizes[-1]))
        
        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        input_shape = input.shape
        if input.dim() == 3:
            output = input[:,None,:,:]
        else:
            output = input.reshape(-1,1, input_shape[2], input_shape[3])
        
        for i in range(1, self.num_layers):
            output = F.conv2d(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)

        logits = F.linear(output.flatten(1),
                          weight=params['layer{0}.weight'.format(self.num_layers)],
                          bias=params['layer{0}.bias'.format(self.num_layers)])

        if input.dim() == 4:
            logits = logits.reshape(input_shape[0], input_shape[1], -1)
        return Categorical(logits=logits)
