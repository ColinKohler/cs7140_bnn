import autograd
import autograd.numpy as np
from autograd.misc.flatten import flatten

from functools import partial

import utils
import nn


import warnings


if __name__ == '__main__':
    # warnings.simplefilter('error', 'RuntimeWarning')

    initializer = np.zeros
    def initializer(args):
        return np.random.randn(*args)
        # return np.random.randn(*args) / np.prod(args)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_mnist()
    n, d = X_train.shape

    sizes = d, 800, 800, 10
    params = nn.new_params(sizes, initializer=initializer)

    activation = nn.relu
    activation_output = nn.norm

    forward = partial(nn.forward,
            activation=activation,
            activation_output=activation_output)

    def loss(params, X, Y, forward):
        output, _ = forward(params, X)
        return -(Y * np.log(output)).sum(axis=1).mean()

    grad = autograd.grad(loss)

    X = X_train
    Y = Y_train
    n_train = X.shape[0]

    idx = np.arange(n)
    def get_batch(X, Y, size):
        np.random.shuffle(idx)
        return X[idx[:size]], Y[idx[:size]]

    clip = 1.
    tcondition, tcount = .2, 10

    count = 0
    while True:
        X_batch, Y_batch = get_batch(X, Y, 100)

        gradients = grad(params, X_batch, Y_batch, forward)
        gnorm = np.sqrt(np.square(flatten(gradients)[0]).sum())
        gclip = clip / gnorm
        for p, g in zip(params, gradients):
            for j in range(2):
                if gclip < 1.:
                    g[j][:] *= gclip

                p[j][:] -= g[j]
                # gnorm += np.square(g[j]).sum()
        l = loss(params, X_batch, Y_batch, forward)
        # output, _ = forward(params, X_batch)
        print(l, gnorm)
        if l < tcondition:
            count += 1
            if count == tcount:
                break
        else:
            count = 0

    output, _ = forward(params, X)

    output_ = output.argmax(axis=1)
    Y_ = Y.argmax(axis=1)
    print('misclassification', (output_ != Y_).sum() / n_train)
