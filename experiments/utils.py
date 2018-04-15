import sklearn.datasets
import sklearn.model_selection

import os
import numpy as np
import numpy.random as npr

# Load mnist dataset with preprocess and train/val/test split detailed in paper
def load_mnist():
    mnist_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home=mnist_path)
    X = mnist.data.astype(np.float32) / 126.
    Y = mnist.target.astype(np.float32)

    train_split = 20000. / mnist.data.shape[0]
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=train_split)
    X_test, X_val, Y_test, Y_val = sklearn.model_selection.train_test_split(X_test, Y_test, test_size=0.5)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# Generate dummy regression data acourding to the following function:
# y = x + 0.3 * sin(2*pi * (x + eps)) + 0.3 * sin(4*pi * (x + eps)) + eps
def generate_regression_curve_data(x_min=0, x_max=0.5, num_samples=1000):
    x = np.linspace(x_min, x_max, num=num_samples)
    eps = npr.normal(0.0, 0.02, size=num_samples)
    y = x + 0.3 * np.sin(2*np.pi * (x + eps)) + 0.3 * np.sin(4*np.pi * (x+eps)) + eps

    return x, y

# Create a batch iterator over the given data
def create_batch_iterator(X, Y, batch_size):
    i = 0
    while i < X.shape[0]:
        X_batch, Y_batch = X[i:i+batch_size], Y[i:i+batch_size]
        i += batch_size
        yield X_batch, Y_batch

# Evaluate the given model's accuracy on the given data
def evaluate_accuracy():
    return 0.
