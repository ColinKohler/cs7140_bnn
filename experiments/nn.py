import autograd
import autograd.numpy as np

import more_itertools as mitt


# logistic activation function
def logistic(x):
    return 1 / (1 + np.exp(-x))


# soft-relu activation function
def relu(x):
    return np.log1p(x)


# initializes and returns new set of parameters
def new_params(sizes, initializer=None):
    if initializer is None:
        initializer = np.zeros

    params = []
    for i, (din, dout) in enumerate(mitt.pairwise(sizes)):
        params.append((
            initializer((dout, din)),
            initializer((dout,)),
        ))

    return params


# forward-propagation:  computes layer values
def forward(params, x, activation, activation_output=None):
    if activation_output is None:
        activation_output = activation

    nlayers = len(params)+1
    layers = np.empty(nlayers, dtype=object)
    layers[0] = x
    for i, p in enumerate(params[:-1]):
        layers[i+1] = activation(p[0] @ layers[i] + p[1])
    layers[-1] = activation_output(params[-1][0] @ layers[-2] + params[-1][1])
    return layers


if __name__ == '__main__':
    # example usage
    from functools import partial


    initializer = None
    initializer = np.ones
    initializer = np.zeros
    def initializer(args):
        return .1 * np.random.randn(*args)

    params = new_params([2, 3, 5, 2], initializer=initializer)
    print('parameters')
    for i, p in enumerate(params):
        print(i, 0, p[0].shape, p[0])
        print(i, 1, p[1].shape, p[1])
    print()

    activation = relu
    activation_output = None
    forward = partial(forward,
            activation=activation,
            activation_output=activation_output)

    x = np.array([1., 2.])
    y = np.array([0., 1.])

    layers = forward(params, x)
    print('layers')
    for i, l in enumerate(layers):
        print(i, l.shape, l)
    print()

    def loss(params, x, y, forward):
        layers = forward(params, x)
        return np.square(layers[-1] - y).sum()

    grad = autograd.grad(loss)
    gradients = grad(params, x, y, forward)
    print('gradients')
    for i, g in enumerate(gradients):
        print(i, 0, g[0].shape, g[0])
        print(i, 1, g[1].shape, g[1])
    print()
