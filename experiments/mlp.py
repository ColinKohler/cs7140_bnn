import autograd
import autograd.numpy as np
import autograd.numpy.random as rnd
from autograd.misc.flatten import flatten

import itertools as itt
import more_itertools as mitt


class MLP:
    def __init__(self, layers, activation, activation_output):
        self.layers = layers
        self.activation = activation
        self.activation_output = activation_output

    #  initializes and returns new set of parameters
    def new_params(self, initializer, initializer2=None):
        return [(initializer((din, dout)), initializer((dout,)))
                for din, dout in mitt.pairwise(self.layers)]

    # takes one gradient step
    def update(self, params, gradients, lr):
        for p, g in zip(itt.chain(*params), itt.chain(*gradients)):
            p[:] -= lr * g

    # forward-propagation:  computes layer values
    def forward(self, params, X):
        *hparams, oparams = params
        Z = np.atleast_2d(X)

        for hpA, hpb in hparams:
            Z_in = Z @ hpA + hpb
            Z = self.activation(Z_in)
        Z_in = Z @ oparams[0] + oparams[1]
        return self.activation_output(Z_in)
