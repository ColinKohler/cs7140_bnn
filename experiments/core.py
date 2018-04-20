import autograd.numpy as np
from autograd.scipy.misc import logsumexp


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
