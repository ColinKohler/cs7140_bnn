import argparse
from functools import partial
import itertools as itt
from tqdm import tqdm, trange

import autograd
import autograd.numpy as np
import autograd.numpy.random as rnd

# Our junk
import utils
import core
import mlp
import bnn


def initializer(args):
    return .05 * rnd.randn(*args)


def loss_mlp(params, model, X, Y):
    output = model.forward(params, X)
    return -(Y * np.log(output)).sum()

def error_mlp(params, model, X, Y):
    ndata, _ = X.shape
    output = model.forward(params, X)
    return (output.argmax(axis=1) != Y.argmax(axis=1)).mean()


def loss_bnn(params, model, X, Y, nsamples=1):
    wsamples = (model.sample_wparams(params) for _ in range(nsamples))
    losses = (loss_mlp(wparams, model.mlp, X, Y) for wparams in wsamples)
    return sum(losses)

def error_bnn(params, model, X, Y, nsamples=1):
    wsamples = (model.sample_wparams(params) for _ in range(nsamples))
    errors = (error_mlp(wparams, model.mlp, X, Y) for wparams in wsamples)
    return sum(errors) / nsamples


def factory(config):
    if config.model == 'mlp':
        modelcls = mlp.MLP
        loss = loss_mlp
        error = error_mlp
    elif config.model == 'bnn':
        modelcls = bnn.BNN
        loss = partial(loss_bnn, nsamples=config.nsamples)
        error = partial(error_bnn, nsamples=config.nsamples)
    else:
        raise ValueError(f'Invalid model name `{config.model}`.')

    return modelcls, loss, error


def run_mnist_classification(config):
    modelcls, loss, error = factory(config)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_mnist()
    N, D = X_train.shape
    _, C = Y_train.shape
    print('N, D, C:', N, D, C)

    layers = [D] + config.hidden_layers + [C]
    activation = core.relu
    activation_output = core.softmax
    model = modelcls(layers, activation, activation_output)
    params = model.new_params(initializer)

    grad = autograd.grad(loss)

    train_errs = list()
    test_errs = list()
    losses = list()
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
    for epoch in range(config.epochs):
        batch_iterator = utils.create_batch_iterator(X_train, Y_train, config.batch_size)
        # print(f'# Epoch:{epoch:>3}')
        with tqdm(total=N) as pbar:
            for i, (X_batch, Y_batch) in enumerate(batch_iterator):
                l = loss(params, model, X_batch, Y_batch)
                gradients = grad(params, model, X_batch, Y_batch)

                pbar.set_description(f'Batch:{i:>3}; Loss:{l:>6.2f}')
                pbar.update(len(X_batch))

                model.update(params, gradients, lr=config.lr)

        l = loss(params, model, X_train, Y_train)
        train_error = error(params, model, X_train, Y_train)
        test_error = error(params, model, X_test, Y_test)

        losses.append(l)
        train_errs.append(train_error)
        test_errs.append(test_error)
        print(f'Total -    Loss:{l:>6.2f}; Train Error:{100*train_error:>5.1f}% \t; Test Error:{100*test_error:>5.1f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['mlp', 'bnn'])
    parser.add_argument('--epochs', type=int, default=100,
            help='Number of epochs to train classifier for')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[400, 400],
            help='Number of neurons in each hidden layer')
    parser.add_argument('--batch_size', type=int, default=128,
            help='Number of samples in a minibatch')
    parser.add_argument('--lr', type=float, default=1e-3,
            help='Learning rate')

    # BNN args
    parser.add_argument('--nsamples', type=int, default=1,
            help='Number of samples used in Bayes by Backprop')
    parser.add_argument('--log_sigma1', type=float, default=.000001,
            help='Log variance for the first gaussian prior on weights')
    parser.add_argument('--log_sigma2', type=float, default=None,
            help='Log variance for the second gaussian prior on weights')
    parser.add_argument('--pi', type=float, default=None,
            help='Amount to weight each gaussian prior in the scale mixture model')

    args = parser.parse_args()
    print('args:', args)

    run_mnist_classification(args)
