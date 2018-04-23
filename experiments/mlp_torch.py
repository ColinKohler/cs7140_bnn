import torch
import torch.nn as nn
import torch.nn.functional as F

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
        for layer in list(self.layers)[:-1]:
            x = self.activation(layer(x))
        x = self.activation_output(self.layers[-1](x), dim=1)
        return x

def loss(model, X, Y, loss_fn):
    output = model(X)
    return [output], loss_fn(output, Y, size_average=False)
