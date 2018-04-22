import torch
import torch.nn as nn

import more_itertools as mitt

class MLP(nn.Module):
    def __init__(self, layer_dims, activation, activation_output):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for d_in, d_out in mitt.pairwise(layer_dims):
            self.layers.append(nn.Linear(d_in, d_out))

        self.activation = activation
        self.activation_output = activation_output

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.activation_output(x, dim=1)
        return x
