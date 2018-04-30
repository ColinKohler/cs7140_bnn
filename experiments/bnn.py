import autograd
import autograd.numpy as np
import autograd.numpy.random as rnd

import itertools as itt
import more_itertools as mitt

import utils
import mlp


class BNN:
    def __init__(self, layers, activation, activation_output):
        self.mlp = mlp.MLP(layers, activation, activation_output)

    def std(self, r):
        return np.log1p(np.exp(r))

    #  initializes and returns new set of parameters
    def new_params(self, initializer, initializer2=None):
        if initializer2 is None:
            initializer2 = initializer
        mparams = self.mlp.new_params(initializer)
        rparams = self.mlp.new_params(initializer2)
        return mparams, rparams

    def sample_wparams(self, params, *, with_llks=False):
        wparams = []
        for (mA, mb), (rA, rb) in zip(*params):
            sA = self.std(rA)
            sb = self.std(rb)
            eA, eb = rnd.randn(*mA.shape), rnd.randn(*mb.shape)
            wA, wb = mA + sA * eA, mb + sb * eb
            wparams.append((wA, wb))

        return wparams

    def logq(self, params, wparams):
        logq = 0.
        for (mA, mb), (rA, rb), (wA, wb) in zip(*params, wparams):
            vA = self.std(rA) ** 2
            vb = self.std(rb) ** 2
            logq -= np.sum(.5 * ((wA-mA)**2 / vA + np.log(2 * np.pi * vA)))
            logq -= np.sum(.5 * ((wb-mb)**2 / vb + np.log(2 * np.pi * vb)))
        return logq


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
