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


def reward(A, Y):
    C = rnd.randint(2, size=A.size)
    R = np.select([1-A, Y, C], [0., 5., 5.], default=-35.)
    return R


def regret(Y, R):
    oracle_R = np.where(Y, 5., 0.)
    return oracle_R - R


def loss_mlp(params, model, X, A, R):
    X[:, 0], X[:, 1] = 1-A, A
    R_ = model.forward(params, X)
    return np.square(R - R_).sum()


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


# TODO insert prior
def loss_bnn(params, model, X, A, R, pi, logs1, logs2, nsamples):
    loss = 0.
    for _ in range(nsamples):
        wparams = model.sample_wparams(params)
        logq = model.logq(params, wparams)
        logp = loss_logp(wparams, pi, logs1, logs2)
        kl = logq - logp
        # print(logq, logp, kl)

        # data likelihood
        loss += loss_mlp(wparams, model.mlp, X, A, R)
        # model prior
        loss += kl
    return loss / nsamples


def action_selection_mlp(params, model, X, actions, epsilon):
    X = np.atleast_2d(X)
    n, _ = X.shape

    nactions = len(actions)
    R = np.empty((nactions, n))
    for ai, a in enumerate(actions):
        X[:, :nactions] = a
        R[ai] = model.forward(params, X).reshape(-1)
    print()
    print('rewards (action_selection):', R.mean(axis=1))
    A = np.argmax(R, axis=0)
    A_ = rnd.choice(2, size=n)
    G = rnd.choice([False, True], p=[epsilon, 1-epsilon], size=n)
    return np.where(G, A, A_)


def action_selection_bnn(params, model, X, actions, nsamples):
    X = np.atleast_2d(X)
    n, _ = X.shape

    nactions = len(actions)
    R = np.empty((nactions, nsamples, n))
    for ai, a in enumerate(actions):
        X[:, :nactions] = a
        for si in range(nsamples):
            R[ai, si] = model.forward(params, X).reshape(-1)
    print()
    print('rewards (action_selection):', R.mean(axis=(1, 2)))
    A = R.mean(axis=1).argmax(axis=0)
    A_ = rnd.choice(2, size=n)
    G = rnd.choice([False, True], size=n)
    return np.where(G, A, A_)


def initializer(args):
    r = np.sqrt(6 / sum(args))
    return rnd.uniform(low=-r, high=r, size=args)


def initializer2(args):
    return initializer(args)-3


def factory(config):
    if config.model == 'mlp':
        modelcls = mlp.MLP
        loss = loss_mlp
        action_selection = partial(action_selection_mlp, epsilon=config.epsilon)
    elif config.model == 'bnn':
        modelcls = bnn.BNN
        loss = partial(loss_bnn,
                       pi=config.pi,
                       logs1=config.neglog_sigma1,
                       logs2=config.neglog_sigma2,
                       nsamples=config.nsamples)
        action_selection = partial(action_selection_bnn, nsamples=config.nsamples)
    else:
        raise ValueError(f'Invalid model name `{config.model}`.')

    return modelcls, loss, action_selection


def run_mushroom_cbandits(config):
    modelcls, loss, action_selection = factory(config)

    X, Y = utils.load_mushroom()
    N, D = X.shape
    print('N, D:', N, D)

    nactions = 2
    actions = np.eye(nactions)

    layers = [D] + config.hidden_layers + [1]
    activation = utils.relu
    activation_output = utils.identity
    model = modelcls(layers, activation, activation_output)
    # params = model.new_params(initializer)
    params = model.new_params(initializer, initializer2)

    grad = autograd.grad(loss)

    # Initializing buffer
    buff_size = config.batch_size ** 2
    idxs = rnd.choice(N, size=buff_size)
    X_buff = X[idxs]
    Y_buff = Y[idxs]
    A_buff = action_selection(params, model, X_buff, actions)
    R_buff = reward(A_buff, Y_buff)

    cumreg = 0
    regrets = []
    with tqdm(itt.count()) as pbar:
        for t in pbar:
            if config.train_batch:
                k = (t * config.batch_size) % buff_size
                buff_idxs = slice(k, k+config.batch_size)
                idxs = rnd.choice(N, size=config.batch_size)
                X_batch = X[idxs]
                Y_batch = Y[idxs]
                A_batch = action_selection(params, model, X_batch, actions)
                R_batch = reward(A_batch, Y_batch)
                G_batch = regret(Y_batch, R_batch)
            else:
                k = t % buff_size
                buff_idxs = slice(k, k+1)
                idxs = rnd.choice(N, size=1)
                X_batch = X[idxs]
                Y_batch = Y[idxs]
                A_batch = action_selection(params, model, X_batch, actions)
                R_batch = reward(A_batch, Y_batch)
                G_batch = regret(Y_batch, R_batch)

            print('rewards():')
            print(' - A:', A_batch.mean())
            print(' - Y:', Y_batch.mean())
            print(' - R:', R_batch.mean())
            print(' - G:', G_batch.mean())

            X_buff[buff_idxs] = X_batch
            Y_buff[buff_idxs] = Y_batch
            A_buff[buff_idxs] = A_batch
            R_buff[buff_idxs] = R_batch

            regrets.append(G_batch.sum())
            cumreg += G_batch.sum()

            batch_iterator = utils.create_batch_iterator(config.batch_size, X_buff, A_buff, R_buff)
            l = 0.
            # l = loss(params, model, X_buff, Y_buff, R_buff)
            for X_batch, A_batch, R_batch in batch_iterator:
                # l += loss(params, model, X_batch, A_batch, R_batch)
                gradients = grad(params, model, X_batch, A_batch, R_batch)

                model.update(params, gradients, lr=config.lr)

            pbar.set_description(f'Epoch:{t:>3}; Loss:{l:>10.2f}; CumReg:{cumreg}')


if __name__ == '__main__':
    np.seterr(all='raise')

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['mlp', 'bnn'])
    parser.add_argument('--epochs', type=int, default=100,
            help='Number of epochs to train classifier for')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[100, 100],
            help='Number of neurons in each hidden layer')
    parser.add_argument('--batch_size', type=int, default=64,
            help='Number of samples in a minibatch')
    parser.add_argument('--lr', type=float, default=1e-5,
            help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.,
            help='Epsilon exploration strategy')

    parser.add_argument('--train_batch', action='store_true',
            help='See whole new batch or single mushroom for each time step')

    # BNN args
    parser.add_argument('--nsamples', type=int, default=2,
            help='Number of samples used in Bayes by Backprop')
    parser.add_argument('--pi', type=float, default=3/4,
            help='Amount to weight each gaussian prior in the scale mixture model')
    parser.add_argument('--neglog_sigma1', type=float, default=1,
            help='Log variance for the first gaussian prior on weights')
    parser.add_argument('--neglog_sigma2', type=float, default=7,
            help='Log variance for the second gaussian prior on weights')

    args = parser.parse_args()
    print('args:', args)

    run_mushroom_cbandits(args)
