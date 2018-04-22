import argparse
from functools import partial
import itertools as itt
from tqdm import tqdm, trange

import autograd
import autograd.numpy as np
import autograd.numpy.random as rnd

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

# Our junk
import utils
import mlp_torch
import bnn_torch

def train(model, loader, N, optimizer, cuda=True):
    model.train()
    with tqdm(total=N) as pbar:
        for i, (x_batch, y_batch) in enumerate(loader):
            if cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            x_batch, y_batch = Variable(x_batch), Variable(y_batch)
            optimizer.zero_grad()

            output = model(x_batch)
            loss = F.nll_loss(output, y_batch)
            loss.backward()
            optimizer.step()

            l = loss.data[0]
            pbar.set_description(f'Batch:{i:>3}; Loss:{l:>6.2f}')
            pbar.update(len(x_batch))

def test(model, loader, cuda=True):
    model.eval()
    loss = 0.
    correct = 0
    for x_batch, y_batch in loader:
        if cuda:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        x_batch, y_batch = Variable(x_batch, volatile=True), Variable(y_batch)

        output = model(x_batch)
        loss += F.nll_loss(output, y_batch, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y_batch.data.view_as(pred)).long().cpu().sum()

    accuracy = 100. * correct / len(loader.dataset)
    loss /= len(loader.dataset)
    return loss, accuracy

def loss_bnn(model, X, Y, nsamples):
    loss = 0.
    for _ in range(nsamples):
        output = model(X)
        loss += F.nll_loss(output, Y).data[0]
    return loss

def factory(conifg):
    if config.model == 'mlp':
        model_class = mlp_torch.MLP
        loss_fn = F.nll_loss
    elif config.model == 'bnn':
        model_class = bnn_torch.BNN
        loss_fn = partial(loss_bnn,
                          nsamples=config.nsamples)
    else:
        raise ValueError(f'Invalid model name `{config.model}`.')

    return model_class, loss_fn

def run_mnist_classification(config):
    N, D, C, train_loader, val_loader, test_loader = utils.load_mnist_torch(config.batch_size)
    print('N, D, C:', N, D, C)

    layers = [D] + config.hidden_layers + [C]
    activation = F.relu
    activation_output = F.log_softmax
    if config.model == 'mlp':
        model = mlp_torch.MLP(layers, activation, activation_output)
    else:
        model = bnn_torch.BNN(layers, activation, activation_output)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    train_errs = list()
    test_errs = list()
    losses = list()
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
    for epoch in range(config.epochs):
        train(model, train_loader, N, optimizer)
        train_loss, train_accuracy = test(model, train_loader)
        test_loss, test_accuracy = test(model, test_loader)

        train_error = 100. - train_accuracy
        test_error = 100. - test_accuracy

        losses.append(test_loss)
        train_errs.append(train_error)
        test_errs.append(test_error)
        print(f'Epoch:{epoch:>3}   Loss:{test_loss:>6.2f}; Train Error:{train_error:>6.2f}% \t; Test Error:{test_error:>6.2f}%')


if __name__ == '__main__':
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
