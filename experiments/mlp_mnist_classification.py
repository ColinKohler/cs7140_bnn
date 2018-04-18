import argparse
from functools import partial

import autograd
import autograd.numpy as np
from autograd.misc.flatten import flatten

# Our junk
import utils
import nn

import itertools as itt


def initializer(args):
    return np.random.randn(*args)

def loss(params, X, Y, forward):
    output, _ = forward(params, X)
    return -(Y * np.log(output)).sum(axis=1).mean()

def run_mnist_classification(config):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_mnist()
    N, D = X_train.shape
    _, C = Y_train.shape

    layer_sizes = [D] + [config.num_hidden_units] * config.num_hidden_layers + [C]
    params = nn.new_params(layer_sizes, initializer=initializer)

    activation = nn.relu
    output = nn.softmax

    forward = partial(nn.forward,
            activation=activation,
            activation_output=output)
    grad = autograd.grad(loss)

    train_errs = list()
    test_errs = list()
    losses = list()
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
    for epoch in range(config.epochs):
        batch_iterator = utils.create_batch_iterator(X_train, Y_train, config.batch_size)
        for X_batch, Y_batch in batch_iterator:
            gradients = grad(params, X_batch, Y_batch, forward)

            if config.clip is not None:
                gnorm2 = np.square(flatten(gradients)[0]).sum()
                if gnorm2 > config.clip ** 2:
                    gclip = config.clip / np.sqrt(gnorm2)
                    for g in itt.chain(*gradients):
                        g[:] *= gclip

            for p, g in zip(itt.chain(*params), itt.chain(*gradients)):
                p[:] -= g

        l = loss(params, X_batch, Y_batch, forward)
        train_error = utils.mnist_error(params, X_train, Y_train, forward)
        test_error = utils.mnist_error(params, X_test, Y_test, forward)

        losses.append(l)
        train_errs.append(train_error)
        test_errs.append(test_error)
        print(f'Epoch:{epoch:>3} \t; Loss:{l:3.5f} \t; Train Error:{100*train_error:>5.1f}% \t; Test Error:{100*test_error:>5.1f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100,
            help='Number of epochs to train classifier for')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
            help='Number of hidden layers')
    parser.add_argument('--num_hidden_units', type=int, default=400,
            help='Number of hidden units per layer')
    parser.add_argument('--batch_size', type=int, default=128,
            help='Number of samples in a minibatch')
    parser.add_argument('--clip', type=float, default=None,
            help='Gradient clipping')
    # TODO lr is unused
    parser.add_argument('--lr', type=float, default=0.1,
            help='Learning rate')

    args = parser.parse_args()
    run_mnist_classification(args)
