import argparse
from functools import partial
import itertools as itt
from tqdm import tqdm, trange
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Our junk
import utils
import mlp_torch
import bnn_torch

# Train model for 1 epoch
def train(model, loader, N, optimizer, loss_fn, cuda=True):
    model.train()
    with tqdm(total=N) as pbar:
        for i, (x_batch, y_batch) in enumerate(loader):
            if cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            x_batch, y_batch = Variable(x_batch), Variable(y_batch)
            optimizer.zero_grad()
            model.zero_grad()

            output, loss = loss_fn(model, x_batch, y_batch)
            loss.backward()
            optimizer.step()

            l = loss.data[0]
            pbar.set_description(f'Batch:{i:>3}; Loss:{l:>6.2f}')
            pbar.update(len(x_batch))

# Test model over the given dataset
def test(model, loader, loss_fn, cuda=True):
    model.eval()
    loss = 0.
    correct = 0
    for x_batch, y_batch in loader:
        if cuda:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        x_batch, y_batch = Variable(x_batch, volatile=True), Variable(y_batch)

        outputs, l = loss_fn(model, x_batch, y_batch)
        loss += l.data[0]
        for output in outputs:
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(y_batch.data.view_as(pred)).long().cpu().sum() / len(outputs))

    accuracy = 100. * correct / len(loader.dataset)
    return loss, accuracy

# Build various model components for either MLP or BNN
def factory(config, nbatches):
    if config.model == 'mlp':
        model_class = mlp_torch.MLP
        loss_fn = partial(mlp_torch.loss,
                          loss_fn=F.nll_loss)
        test_fn = test
    elif config.model == 'bnn':
        model_class = partial(bnn_torch.BNN,
                              pi=config.pi,
                              logsigma1=np.exp(-config.neglog_sigma1),
                              logsigma2=np.exp(-config.neglog_sigma2))
        loss_fn = partial(bnn_torch.loss,
                          loss_fn=F.nll_loss,
                          nsamples=config.nsamples,
                          nbatches=nbatches)
        test_fn = test
    else:
        raise ValueError(f'Invalid model name `{config.model}`.')

    return model_class, loss_fn, test_fn

# Main loop: calls train/test over a number of epochs
def run_mnist_classification(config):
    N, D, C, train_loader, val_loader, test_loader = utils.load_mnist_torch(config.batch_size)
    print('N, D, C:', N, D, C)

    model_class, loss_fn, test_fn = factory(config, len(train_loader))
    layers = [D] + config.hidden_layers + [C]
    activation = F.relu
    activation_output = F.log_softmax
    model = model_class(layers, activation, activation_output)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    train_errs = list()
    test_errs = list()
    losses = list()
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
    for epoch in range(config.epochs):
        train(model, train_loader, N, optimizer, loss_fn)
        train_loss, train_accuracy = test_fn(model, train_loader, loss_fn)
        test_loss, test_accuracy = test_fn(model, test_loader, loss_fn)

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
