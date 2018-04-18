import autograd
import autograd.numpy as np
from autograd.scipy.misc import logsumexp

import more_itertools as mitt


# logistic activation function
def logistic(x):
    return 1 / (1 + np.exp(-x))


# soft-relu activation function
def relu(x):
    return np.exp(x - np.logaddexp(0, x))
    # return np.log1p(x)


# softmax output actionation
def softmax(x):
    return np.exp(x - logsumexp(x, axis=1, keepdims=True))


# initializes and returns new set of parameters
def new_params(sizes, initializer):
    return [(initializer((din, dout)), initializer((dout,)))
            for din, dout in mitt.pairwise(sizes)]


# forward-propagation:  computes layer values
def forward(params, X, activation, activation_output=None):
    if activation_output is None:
        activation_output = activation

    layers = [None] * len(params)
    layers[0] = X
    for i, p in enumerate(params[:-1]):
        layers[i + 1] = activation(layers[i] @ p[0] + p[1])
    output = activation_output(layers[-1] @ params[-1][0] + params[-1][1])
    return output, layers


if __name__ == '__main__':
    # example usage
    from functools import partial

    # initializer = np.ones
    # initializer = np.zeros
    def initializer(args):
        return .1 * np.random.randn(*args)

    # params = new_params([2, 3, 5, 2], initializer=initializer)
    params = new_params([2, 10, 2], initializer=initializer)
    print('parameters')
    for i, p in enumerate(params):
        print(i, 0, p[0].shape, p[0])
        print(i, 1, p[1].shape, p[1])
    print()

    activation = relu
    activation_output = None
    forward = partial(
        forward,
        activation=activation,
        activation_output=activation_output,
    )

    x = np.array([1., 2.])
    y = np.array([0., 1.])

    output, layers = forward(params, x)
    print('layers')
    for i, l in enumerate(layers):
        print(i, l.shape, l)
    print('out', output.shape, output)
    print()

    def loss(params, x, y, forward):
        output, _ = forward(params, x)
        return np.square(output - y).sum()

    grad = autograd.grad(loss)
    gradients = grad(params, x, y, forward)
    print('gradients')
    for i, g in enumerate(gradients):
        print(i, 0, g[0].shape, g[0])
        print(i, 1, g[1].shape, g[1])
    print()

    for i in range(10):
        gradients = grad(params, x, y, forward)
        gnorm = 0.
        for p, g in zip(params, gradients):
            for j in range(2):
                p[j][:] += 1 * g[j]
                gnorm += np.square(g[j]).sum()
        print(loss(params, x, y, forward), gnorm)
