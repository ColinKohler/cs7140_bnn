import os
import numpy as np
import numpy.random as npr
from sklearn import utils, datasets, model_selection

# Load mnist dataset with preprocess and train/val/test split detailed in paper
def load_mnist():
    mnist_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    mnist = datasets.fetch_mldata('MNIST original', data_home=mnist_path)
    X = mnist.data.astype(np.float32) / 126.
    Y = mnist.target.astype(np.int32)

    # converts labels into "indicator" vectors
    I = np.eye(10, dtype=np.int32)
    Y = I[Y]

    train_split = 20000. / mnist.data.shape[0]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=train_split)
    X_test, X_val, Y_test, Y_val = model_selection.train_test_split(X_test, Y_test, test_size=0.5)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# Generate dummy regression data acourding to the following function:
# y = x + 0.3 * sin(2*pi * (x + eps)) + 0.3 * sin(4*pi * (x + eps)) + eps
def generate_regression_curve_data(x_min=0, x_max=0.5, num_samples=1000):
    x = np.linspace(x_min, x_max, num=num_samples)
    eps = npr.normal(0.0, 0.02, size=num_samples)
    y = x + 0.3 * np.sin(2*np.pi * (x + eps)) + 0.3 * np.sin(4*np.pi * (x+eps)) + eps

    return x, y

# Find the L1 or L2 norm between the predicted point and the true point
def regression_error(x, y_pred, norm='L2'):
    y_true = x + 0.3 * np.sin(2*np.pi*x) + 0.3 * np.sin(4*np.pi*x)

    if norm == 'L2':
        return np.linalg.norm(y_pred - y_true)
    elif norm == 'L1':
        return np.abs(y_pred - y_true)

def create_batch_iterator(X, Y, batch_size):
    X, Y = utils.shuffle(X,Y)

    i = 0
    while i < X.shape[0]:
        X_batch, Y_batch = X[i:i+batch_size], Y[i:i+batch_size]
        i += batch_size
        yield X_batch, Y_batch

# Evaluate the given model's error on the given data
def mnist_error(params, X, Y, forward):
    n, _ = X.shape
    output, _ = forward(params, X)
    return (output.argmax(axis=1) != Y.argmax(axis=1)).sum() / n
