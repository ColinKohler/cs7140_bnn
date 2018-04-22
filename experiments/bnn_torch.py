import torch
import torch.nn as nn
from torch.autograd import Variable

import more_itertools as mitt

class BNN_Layer(nn.Module):
    def __init__(self, d_input, d_output):
        super(BNN_Layer, self).__init__()
        self.d_input = d_input
        self.d_output = d_output

        # Create weight and bias parameters for mu and rho
        self.W_m = nn.Parameter(torch.Tensor(self.d_input, self.d_output))
        self.W_r = nn.Parameter(torch.Tensor(self.d_input, self.d_output))
        self.b_m = nn.Parameter(torch.Tensor(self.d_output))
        self.b_r = nn.Parameter(torch.Tensor(self.d_output))

        # Initializer the weights (TODO: This is probably bad)
        self.W_m.data.normal_(0, 0.1)
        self.W_r.data.normal_(0, 0.1)
        self.b_m.data.uniform_(-0.1, 0.1)
        self.b_r.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        eps_W = Variable(torch.Tensor(self.d_input, self.d_output).normal_(0,1)).cuda()
        eps_b = Variable(torch.Tensor(self.d_output).normal_(0,1)).cuda()
        sigma_W = torch.log1p(torch.exp(self.W_r))
        sigma_b = torch.log1p(torch.exp(self.b_r))

        W = self.W_m + sigma_W * eps_W
        b = self.b_m + sigma_b * eps_b

        return torch.mm(x, W) + b.expand(x.size()[0], self.d_output)

class BNN(nn.Module):
    def __init__(self, layer_dims, activation, activation_output):
        super(BNN, self).__init__()
        self.layers = nn.ModuleList()
        for d_in, d_out in mitt.pairwise(layer_dims):
            self.layers.append(BNN_Layer(d_in, d_out))

        self.activation = activation
        self.activation_output = activation_output

    def forward(self, x):
        for layer in list(self.layers)[:-1]:
            x = self.activation(layer(x))
        x = self.activation_output(self.layers[-1](x), dim=1)
        return x
