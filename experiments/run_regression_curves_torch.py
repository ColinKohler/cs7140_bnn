import argparse
from functools import partial
import itertools as itt
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
    l2_norm = 0
    all_outputs = list()
    all_inputs = list()
    for x_batch, y_batch in loader:
        if cuda:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        x_batch, y_batch = Variable(x_batch, volatile=True), Variable(y_batch)

        outputs, l = loss_fn(model, x_batch, y_batch)
        loss += l.data[0]
        for output in outputs:
            l2_norm += np.linalg.norm(y_batch.data.cpu() - output.data.cpu()).sum() / len(outputs)
            all_outputs.append(output.data.cpu().numpy())
            all_inputs.append(x_batch)

    return all_inputs, all_outputs, loss, l2_norm / len(loader)

# Build various model components for either MLP or BNN
def factory(config, nbatches):
    if config.model == 'mlp':
        model_class = mlp_torch.MLP
        loss_fn = partial(mlp_torch.loss,
                          loss_fn=F.smooth_l1_loss)
        test_fn = test
    elif config.model == 'bnn':
        model_class = partial(bnn_torch.BNN,
                              pi=config.pi,
                              sigma1=0.75,
                              sigma2=0.1)
        loss_fn = partial(bnn_torch.loss,
                          loss_fn=F.smooth_l1_loss,
                          nsamples=config.nsamples,
                          nbatches=nbatches)
        test_fn = test
    else:
        raise ValueError(f'Invalid model name `{config.model}`.')

    return model_class, loss_fn, test_fn

# Main loop: calls train/test over a number of epochs
def run_regression_curves(config):
    N, D, C, train_loader, val_loader, test_loader = utils.load_regression_curve_torch(config.batch_size)
    extended_loader = utils.load_regression_curve_extended_torch(num_samples=config.batch_size, rep=10)
    print('N, D, C:', N, D, C)

    model_class, loss_fn, test_fn = factory(config, len(train_loader))
    layers = [D] + config.hidden_layers + [C]
    activation = F.relu
    activation_output = utils.identity
    model = model_class(layers, activation, activation_output)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    train_norms = list()
    test_norms = list()
    losses = list()
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
    for epoch in range(config.epochs):
        train(model, train_loader, N, optimizer, loss_fn)
        inputs, outputs, train_loss, train_norm = test_fn(model, train_loader, loss_fn)
        inputs, outputs, test_loss, test_norm = test_fn(model, test_loader, loss_fn)

        losses.append(test_loss)
        train_norms.append(train_norm)
        test_norms.append(test_norm)
        print(f'Epoch:{epoch:>3}   Loss:{test_loss:>6.2f}; Train L2 Norm:{train_norm:>6.2f} \t; Test L2 Norm:{test_norm:>6.2f}')

    extended_loader1 = utils.load_regression_curve_extended_torch(x_min=0.0, x_max=0.5, num_samples=config.batch_size, rep=1)
    for _ in range(5):
        inputs, outputs, test_loss, test_norm = test_fn(model, extended_loader1, loss_fn)
        plt.scatter(np.array(inputs).flatten(), np.array(outputs).flatten())
        plt.show()

    for _ in range(5):
        inputs, outputs, test_loss, test_norm = test_fn(model, extended_loader, loss_fn)
        plt.scatter(np.array(inputs).flatten(), np.array(outputs).flatten())
        plt.show()

    #for layer in model.layers:
    #    show_weight_dist(layer.W_m.data.cpu().numpy()[0,0], np.log1p(np.exp(layer.W_r.data.cpu().numpy()[0,0])))
    #    show_weight_dist(layer.b_m.data.cpu().numpy()[0], np.log1p(np.exp(layer.b_r.data.cpu().numpy()[0])))
    #    print(f'Wight Mean: `{layer.W_m.data}`;    Bias Mean: `{layer.b_m.data}`')
    #    print(f'Wight Ro: `{layer.W_r.data}`;    Bias Ro: `{layer.b_r.data}`')

def show_weight_dist(mean, variance):
    sigma = np.sqrt(variance)
    x = np.linspace(mean - 4*sigma, mean + 4*sigma, 100)
    plt.plot(x, norm.pdf(x, mean, sigma))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['mlp', 'bnn'])
    parser.add_argument('--epochs', type=int, default=100,
            help='Number of epochs to train classifier for')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[10, 10],
            help='Number of neurons in each hidden layer')
    parser.add_argument('--batch_size', type=int, default=100,
            help='Number of samples in a minibatch')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Learning rate')

    # BNN args
    parser.add_argument('--nsamples', type=int, default=1,
            help='Number of samples used in Bayes by Backprop')
    parser.add_argument('--pi', type=float, default=1/2,
            help='Amount to weight each gaussian prior in the scale mixture model')
    parser.add_argument('--neglog_sigma1', type=float, default=1,
            help='Log variance for the first gaussian prior on weights')
    parser.add_argument('--neglog_sigma2', type=float, default=7,
            help='Log variance for the second gaussian prior on weights')

    args = parser.parse_args()
    print('args:', args)

    run_regression_curves(args)
