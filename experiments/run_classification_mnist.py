import argparse
from functools import partial
import itertools as itt
from tqdm import tqdm, trange

import autograd
import autograd.numpy as np
import autograd.numpy.random as rnd

# Our junk
import utils
import mlp
import bnn


def initializer(args):
    r = np.sqrt(6 / sum(args))
    return rnd.uniform(low=-r, high=r, size=args)


def initializer2(args):
    return initializer(args) - 5


def loss_mlp(params, model, X, Y):
    output = model.forward(params, X)
    return -np.sum(Y * np.log(output))

def error_mlp(params, model, X, Y):
    ndata, _ = X.shape
    output = model.forward(params, X)
    return (output.argmax(axis=1) != Y.argmax(axis=1)).mean()

def loss_logp(wparams, pi, nlogs1, nlogs2):
    log2pi = np.log(2 * np.pi)
    s1 = np.exp(-nlogs1)
    s2 = np.exp(-nlogs2)

    logp = 0.
    for wA, wb in wparams:
        # logp += np.log(
        #     pi * np.exp(- .5 * log2pi + nlogs1 -.5 * (wA/s1)**2)
        #     + (1-pi) * np.exp(- .5 * log2pi + nlogs2 -.5 * (wA/s2)**2)
        # ).sum()
        # logp += np.log(
        #     pi * np.exp(- .5 * log2pi + nlogs1 -.5 * (wb/s1)**2)
        #     + (1-pi) * np.exp(- .5 * log2pi + nlogs2 -.5 * (wb/s2)**2)
        # ).sum()

        # NOTE unstable in autograd.numpy
        # logp += np.logaddexp(
        #     np.log(pi) - .5 * log2pi + nlogs1 -.5 * (wA/s1)**2,
        #     np.log(1-pi) - .5 * log2pi + nlogs2 -.5 * (wA/s2)**2,
        # ).sum()
        # logp += np.logaddexp(
        #     np.log(pi) - .5 * log2pi + nlogs1 -.5 * (wb/s1)**2,
        #     np.log(1-pi) - .5 * log2pi + nlogs2 -.5 * (wb/s2)**2,
        # ).sum()

        logp += np.sum(-.5 * log2pi + nlogs1 -.5 * (wA/s1)**2)
        logp += np.sum(-.5 * log2pi + nlogs1 -.5 * (wb/s1)**2)
    return logp


def loss_bnn(params, model, X, Y, pi, logs1, logs2, nsamples):
    loss = 0.
    for _ in range(nsamples):
        wparams = model.sample_wparams(params)
        logq = model.logq(params, wparams)
        logp = loss_logp(wparams, pi, logs1, logs2)
        kl = logq - logp
        # print(logq, logp, kl)

        # data likelihood
        loss += loss_mlp(wparams, model.mlp, X, Y)
        # model prior
        loss += kl
    return loss / nsamples


#     wsamples = (model.sample_wparams(params) for _ in range(nsamples))
#     losses = (loss_mlp(wparams, model.mlp, X, Y) for wparams in wsamples)
#     return sum(losses)

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

        def initializer2(args):
            return initializer(args)-100

        modelcls = bnn.BNN
        loss = partial(loss_bnn,
                       pi=config.pi,
                       logs1=config.neglog_sigma1,
                       logs2=config.neglog_sigma2,
                       nsamples=config.nsamples)
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
    activation = utils.relu
    # activation = utils.logistic
    activation_output = utils.softmax
    model = modelcls(layers, activation, activation_output)
    params = model.new_params(initializer, initializer2)

    grad = autograd.grad(loss)

    train_errs = list()
    test_errs = list()
    losses = list()
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
    for epoch in range(config.epochs):
        batch_iterator = utils.create_batch_iterator(config.batch_size, X_train, Y_train)
        # print(f'# Epoch:{epoch:>3}')
        with tqdm(total=N) as pbar:
            for i, (X_batch, Y_batch) in enumerate(batch_iterator):
                l = loss(params, model, X_batch, Y_batch)
                gradients = grad(params, model, X_batch, Y_batch)

                pbar.set_description(f'Batch:{i:>3}; Loss:{l:>10.2f}')
                pbar.update(len(X_batch))

                model.update(params, gradients, lr=config.lr)

        l = loss(params, model, X_train, Y_train)
        train_error = error(params, model, X_train, Y_train)
        test_error = error(params, model, X_test, Y_test)

        losses.append(l)
        train_errs.append(train_error)
        test_errs.append(test_error)
        print(f'Epoch:{epoch:>3}; Loss:{l:>10.2f}; Train Error:{100*train_error:>6.2f}%; Test Error:{100*test_error:>6.2f}%')


if __name__ == '__main__':
    np.seterr(all='raise')

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['mlp', 'bnn'])
    parser.add_argument('--epochs', type=int, default=100,
            help='Number of epochs to train classifier for')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[800, 800],
            help='Number of neurons in each hidden layer')
    parser.add_argument('--batch_size', type=int, default=128,
            help='Number of samples in a minibatch')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Learning rate')

    # BNN args
    parser.add_argument('--nsamples', type=int, default=5,
            help='Number of samples used in Bayes by Backprop')
    parser.add_argument('--pi', type=float, default=1/2,
            help='Amount to weight each gaussian prior in the scale mixture model')
    parser.add_argument('--neglog_sigma1', type=float, default=1,
            help='Log variance for the first gaussian prior on weights')
    parser.add_argument('--neglog_sigma2', type=float, default=7,
            help='Log variance for the second gaussian prior on weights')

    args = parser.parse_args()
    print('args:', args)

    run_mnist_classification(args)
