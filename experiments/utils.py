import os
from sklearn import utils, datasets, model_selection
from autograd.scipy.misc import logsumexp
import autograd.numpy as np
import autograd.numpy.random as npr
import torch
import torch.utils.data
from numbers import Number

# Load mnist dataset with preprocess and train/val/test split detailed in paper
def load_mnist():
    mnist_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    mnist = datasets.fetch_mldata('MNIST original', data_home=mnist_path)
    X = mnist.data.astype(np.float32) / 126.
    Y = mnist.target.astype(np.uint8)

    # X = (X-X.mean()) / X.std()
    # X /= X.max()

    # converts labels into "indicator" vectors
    I = np.eye(10, dtype=np.float32)
    Y = I[Y]

    train_split = 20000. / mnist.data.shape[0]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=train_split)
    X_test, X_val, Y_test, Y_val = model_selection.train_test_split(X_test, Y_test, test_size=0.5)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_mnist_torch(batch_size):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist()
    N, D = X_train.shape
    C = 10

    X_train, Y_train = torch.from_numpy(X_train), torch.LongTensor(Y_train)
    X_val, Y_val = torch.from_numpy(X_val), torch.LongTensor(Y_val)
    X_test, Y_test = torch.from_numpy(X_test), torch.LongTensor(Y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True)

    return N, D, C, train_loader, val_loader, test_loader


# Generate dummy regression data acourding to the following function:
# y = x + 0.3 * sin(2*pi * (x + eps)) + 0.3 * sin(4*pi * (x + eps)) + eps
def generate_regression_curve_data(x_min=0, x_max=0.5, num_samples=5000):
    x = np.linspace(x_min, x_max, num=num_samples)
    eps = npr.normal(0.0, 0.02, size=num_samples)
    #y = 2*(x+eps) + 5
    #y_true = 0
    y = x + 0.3 * np.sin(2*np.pi * (x + eps)) + 0.3 * np.sin(4*np.pi * (x+eps)) + eps
    y_true = x + 0.3 * np.sin(2*np.pi*x) + 0.3 * np.sin(4*np.pi*x)
    #y = (x+eps)**2 / 2
    #y_true = x**2 / 2

    return x, y, y_true


def load_mushroom():
    with open('datasets/agaricus-lepiota.data') as f:
        data = np.array([line.rstrip().split(',') for line in f.readlines()])

    n, d = data.shape

    fields = [list(set(data[:, di])) for di in range(d)]
    e = sum(len(f) for f in fields)
    Z = np.zeros((n, e))

    offset = 0
    for di in range(d):
        field = list(set(data[:, di]))
        for ni in range(n):
            ei = offset + field.index(data[ni, di])
            Z[ni, ei] = 1
        offset += len(field)
    # NOTE Y is 1 when edible
    Y = Z[:, 0].astype(np.int32)
    return Z, Y


import more_itertools as mitt


def create_batch_iterator(batch_size, *data):
    data = utils.shuffle(*data)
    ndata = data[0].shape[0]
    for idxs in mitt.chunked(range(ndata), batch_size):
        yield (X[idxs] for X in data)


###################################################################################
#########                      Numpy NN Stuff                            ##########
###################################################################################
# logistic activation function
def logistic(x):
    return 1 / (1 + np.exp(-x))

# relu activation function
def relu(x):
    return np.maximum(0, x)

# soft-relu activation function
def softplus(x):
    return np.exp(x - np.logaddexp(0, x))
    # return np.log1p(x)

# softmax output actionation
def softmax(x):
    return np.exp(x - logsumexp(x, axis=1, keepdims=True))


def identity(x):
    return x

# Simple identity activation function
def identity(x):
    return x

###################################################################################
#########                       PyTorch Stuff                            ##########
###################################################################################
def load_mnist_torch(batch_size):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist()
    N, D = X_train.shape
    C = 10

    X_train, Y_train = torch.from_numpy(X_train), torch.LongTensor(Y_train)
    X_val, Y_val = torch.from_numpy(X_val), torch.LongTensor(Y_val)
    X_test, Y_test = torch.from_numpy(X_test), torch.LongTensor(Y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return N, D, C, train_loader, val_loader, test_loader

def load_regression_curve_torch(batch_size):
    X, Y, Y_true = generate_regression_curve_data(x_min=0.0, x_max=0.5, num_samples=80000)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
    X_test, X_val, Y_test, Y_val = model_selection.train_test_split(X_test, Y_test, test_size=0.5)
    N, D, C = X_train.shape[0], 1, 1

    X_train, Y_train = torch.Tensor(X_train).view(-1,1), torch.Tensor(Y_train).view(-1,1)
    X_val, Y_val = torch.Tensor(X_val).view(-1,1), torch.Tensor(Y_val).view(-1,1)
    X_test, Y_test = torch.Tensor(X_test).view(-1,1), torch.Tensor(Y_test).view(-1,1)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True)

    return N, D, C, train_loader, val_loader, test_loader

def load_regression_curve_extended_torch(x_min=-0.5, x_max=1.0, num_samples=1000, rep=3):
    if rep == 1:
        X, Y, Y_true = generate_regression_curve_data(x_min=x_min, x_max=x_max, num_samples=num_samples)
    else:
        Xs, Ys = list(), list()
        for _ in range(rep):
            X, Y, Y_true = generate_regression_curve_data(x_min=x_min, x_max=x_max, num_samples=num_samples)
            Xs.append(X)
            Ys.append(Y)
        X, Y = np.stack(Xs).flatten(), np.stack(Ys).flatten()
    X, Y = torch.Tensor(X).view(-1,1), torch.Tensor(Y).view(-1,1)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    return loader

# Logsumexp function for PyTroch
def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# TODO should this and/or the loss be out or in models?
# # Evaluate the given model's error on the given data
# def mnist_error(params, X, Y, forward):
#     n, _ = X.shape
#     output, _ = forward(params, X)
#     return (output.argmax(axis=1) != Y.argmax(axis=1)).sum() / n
