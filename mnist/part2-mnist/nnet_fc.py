#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model

def main():
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    model = nn.Sequential(
              nn.Linear(784, 128),
              nn.LeakyReLU(),
              nn.Linear(128, 10)
            )
    lr=0.1
    momentum=0
    ##################################

    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum,n_epochs=10)

    ## Evaluate the model on dev data
    val_loss, val_accuracy = run_epoch(dev_batches, model.eval(), None)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)
    print("Loss on val set:" + str(val_loss) + " Accuracy on val set: " + str(val_accuracy))
    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
## 1
## base - 0.9204727564102564 val 0.9324866310160428
## 64   - 0.9298878205128205 val 0.9398521505376344
## 0.01 - 0.9206730769230769 val 0.9344919786096256
## mom  - 0.8865184294871795 val 0.8935494652406417
## leaky - 0.9207732371794872 val 0.9319852941176471

## 2
## base - val 0.9781082887700535
## 64   - val 0.9768145161290323
## 0.01 - val 0.9550467914438503
## mom  - val 0.9625668449197861
## leaky - val 0.9794451871657754