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

    layer_sizes = [D] + [config.num_hidden_units] * config.num_hidden_layers + [10]
    params = nn.new_params(layer_sizes, initializer=initializer)

    activation = nn.relu
    output = nn.softmax

    forward = partial(nn.forward,
            activation=activation,
            activation_output=output)
    grad = autograd.grad(loss)

    train_acc = list()
    test_acc = list()
    loss_acc = list()
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
        train_accuracy = utils.evaluate_accuracy()
        test_accuracy = utils.evaluate_accuracy()

        loss_acc.append(l)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        print('Epoch:{} Loss:{} Train Accuracy:{} Test Accuracy:{}'
                .format(epoch, l, train_accuracy, test_accuracy))

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
