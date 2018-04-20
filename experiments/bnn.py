import autograd
import autograd.numpy as np
import autograd.numpy.random as rnd

import itertools as itt
import more_itertools as mitt

import mlp


class BNN:
    def __init__(self, layers, activation, activation_output):
        self.mlp = mlp.MLP(layers, activation, activation_output)

    #  initializes and returns new set of parameters
    def new_params(self, initializer):
        mparams = self.mlp.new_params(initializer)
        sparams = self.mlp.new_params(initializer)
        return mparams, sparams

    def sample_wparams(self, params):
        wparams = []
        for (mA, mb), (sA, sb) in zip(*params):
            eA, eb = rnd.randn(*mA.shape), rnd.randn(*mb.shape)
            eA *= .01
            eb *= .01
            vA, vb = np.logaddexp(0, sA), np.logaddexp(0, sb)
            wA, wb = mA + vA * eA, mb + vb * eb
            wparams.append((wA, wb))
        return wparams

    # takes one gradient step
    def update(self, params, gradients, lr):
        # TODO do smth here?
        for (pm, ps), (gm, gs) in zip(itt.chain(*params), itt.chain(*gradients)):
            pm[:] -= lr * gm
            ps[:] -= lr * gs

    def forward(self, params, X):
        wparams = self.sample_wparams(params)
        output = self.mlp.forward(wparams, X)
        return output
