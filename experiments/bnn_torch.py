import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd
import torch.nn.functional as F

import numpy as np
import more_itertools as mitt

from utils import logsumexp

class BNN_Layer(nn.Module):
    def __init__(self, d_input, d_output, pi, sigma1, sigma2):
        super(BNN_Layer, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.layer_num = layer_num

        # Create weight and bias parameters for mu and rho
        self.W_m = nn.Parameter(torch.Tensor(self.d_input, self.d_output))
        self.W_r = nn.Parameter(torch.Tensor(self.d_input, self.d_output))
        self.b_m = nn.Parameter(torch.Tensor(self.d_output))
        self.b_r = nn.Parameter(torch.Tensor(self.d_output))

        # Initialize gaussian/scale prior components
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        # Initialize log probabilities for p and q dists
        self.log_p, self.log_q = 0., 0.

        # Initializer the weights (TODO: This is probably bad)
        self.W_m.data.normal_(0, 0.1)
        self.W_r.data.normal_(-3, 0.01)
        self.b_m.data.normal_(0, 0.1)
        self.b_r.data.normal_(-3, 0.01)

    def forward(self, x):
        # Calculate weight and bais values from randomly sampled epsilons
        sigma_W = torch.log1p(torch.exp(self.W_r))
        sigma_b = torch.log1p(torch.exp(self.b_r))
        W, b = self.sample_weights()

        # Update log probabilities for p and q dists
        self.log_p = self._calculate_log_p(W, b)
        self.log_q = self._calculate_log_q(sigma_W, sigma_b, W, b)

        # Return output based on sampled parameters
        output = torch.mm(x, W) + b.expand(x.size()[0], self.d_output)
        return output

    def sample_weights(self):
        eps_W = Variable(torch.Tensor(self.d_input, self.d_output).normal_(0,1)).cuda()
        eps_b = Variable(torch.Tensor(self.d_output).normal_(0,1)).cuda()
        sigma_W = torch.log1p(torch.exp(self.W_r))
        sigma_b = torch.log1p(torch.exp(self.b_r))

        W = self.W_m + sigma_W * eps_W
        b = self.b_m + sigma_b * eps_b

        return W, b

    #def _calculate_log_p(self, W, b):
    #    return self._calculate_log_gaussian(W, 0, self.sigma1).sum() + \
    #           self._calculate_log_gaussian(b, 0, self.sigma1).sum()
    def _calculate_log_p(self, W, b):
        log_p = 0.
        log_p += logsumexp(
                torch.stack(
                    [float(np.log(self.pi)) + self._calculate_log_gaussian(W, 0, self.sigma1),
                     float(np.log(1 - self.pi)) + self._calculate_log_gaussian(W, 0, self.sigma2)])).sum()
        log_p += logsumexp(
                torch.stack(
                    [float(np.log(self.pi)) + self._calculate_log_gaussian(b, 0, self.sigma1),
                     float(np.log(1 - self.pi)) + self._calculate_log_gaussian(b, 0, self.sigma2)])).sum()
        return log_p

    def _calculate_log_q(self, sigma_W, sigma_b, W, b):
        return self._calculate_log_gaussian(W, self.W_m, sigma_W).sum() + \
               self._calculate_log_gaussian(b, self.b_m, sigma_b).sum()

    def _calculate_log_gaussian(self, x, mu, sigma):
        return float(-0.5 * np.log(2 * np.pi)) - torch.log(sigma) - (x - mu)**2 / (2 * sigma**2)

class BNN(nn.Module):
    def __init__(self, layer_dims, activation, activation_output, pi, sigma1, sigma2):
        super(BNN, self).__init__()
        self.layers = nn.ModuleList()
        self.sigma1 = Variable(torch.Tensor([float(sigma1)]), requires_grad=False).cuda()
        self.sigma2 = Variable(torch.Tensor([float(sigma2)]), requires_grad=False).cuda()
        for d_in, d_out in mitt.pairwise(layer_dims):
            self.layers.append(BNN_Layer(d_in, d_out, pi, self.sigma1, self.sigma2))

        self.activation = activation
        self.activation_output = activation_output

    # Forward pass over the network
    def forward(self, x):
        for i,layer in enumerate(list(self.layers)[:-1]):
            x = self.activation(layer(x))

        x = self.activation_output(self.layers[-1](x))
        #x = self.layers[-1](x)
        return x

    # Get log p and log q dists by summing over each layer
    def get_log_pq(self):
        log_p, log_q = 0., 0.
        for layer in self.layers:
            log_p += layer.log_p
            log_q += layer.log_q

        return log_p, log_q

# Loss function for BNN roughly: KL - likelihood
def loss(model, X, Y, loss_fn, nsamples, nbatches):
    # Torch Variables to hold data across samples
    log_p_sum = Variable(torch.zeros(1), requires_grad=True).cuda()
    log_q_sum = Variable(torch.zeros(1), requires_grad=True).cuda()
    neg_log_likelihood_sum = Variable(torch.zeros(1), requires_grad=True).cuda()
    outputs = list()

    # Average the various loss components of n samples
    for i in range(nsamples):
        output = model(X)
        log_p, log_q = model.get_log_pq()
        neg_log_likelihood = loss_fn(output, Y, size_average=False)

        log_p_sum += log_p
        log_q_sum += log_q
        neg_log_likelihood_sum += neg_log_likelihood
        outputs.append(output)

    avg_log_p = log_p_sum / nsamples
    avg_log_q = log_q_sum / nsamples
    avg_neg_log_likelihood = neg_log_likelihood_sum / nsamples

    # Loss
    KL = (avg_log_q - avg_log_p)
    loss = ((KL / float(nbatches)) + avg_neg_log_likelihood) / 128.

    return outputs, loss
