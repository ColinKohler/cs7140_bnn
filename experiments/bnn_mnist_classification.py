import argparse
import numpy as np

# Our junk
import utils

def run_mnist_classification(config):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_mnist()
    batch_iterator = utils.create_batch_iterator(X_train, Y_train, config.batch_size)

    train_acc = list()
    test_acc = list()
    np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})
    for epoch in range(config.epochs):
        for X_batch, Y_batch in batch_iterator:
            # Train model
            continue

        loss = 0.
        train_accuracy = utils.evaluate_accuracy()
        test_accuracy = utils.evaluate_accuracy()

        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        print('Epoch:{} Loss:{} Train Accuracy:{} Test Accuracy{}'
                .format(epoch, loss, train_accuracy, test_accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100,
            help='Number of epochs to train classifier for')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
            help='Number of hidden layers')
    parser.add_argument('--num_hidden_units', type=int, default=400,
            help='Number of hidden units per layer')
    parser.add_argument('--batch_size', type=int, default=128,
            help='Number of samples in a minibatch')
    parser.add_argument('--lr', type=float, default=0.1,
            help='Learning rate')
    parser.add_argument('--log_sigma1', type=float, default=1,
            help='Log variance for the first gaussian prior on weights')
    parser.add_argument('--log_sigma2', type=float, default=None,
            help='Log variance for the second gaussian prior on weights')
    parser.add_argument('--pi', type=float, default=None,
            help='Amount to weight each gaussian prior in the scale mixture model')

    args = parser.parse_args()
    run_mnist_classification(args)
