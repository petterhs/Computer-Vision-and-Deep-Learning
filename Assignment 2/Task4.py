## 4a ##

#Just change the hyperparameter HIDDEN_LAYER_NODES from 64 to 32

## 4b ##

#Same as above just use 128 this time


## 4c ##
import numpy as np
import random
import matplotlib.pyplot as plt
import mnist

from Network2 import Network

def train_val_split(X, Y, val_percentage):
    dataset_size = X.shape[0]
    idx = np.arange(0, dataset_size)
    np.random.shuffle(idx)

    train_size = int(dataset_size*(1-val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    return X_train, Y_train, X_val, Y_val

def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot

def bias_trick(X):
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)


def main():
    X_train, Y_train, X_test, Y_test = mnist.load()

    # 2a
    #First 60 000
    NUMBER_OF_TRAINING_DATA = 60000
    X_train = X_train[:NUMBER_OF_TRAINING_DATA]
    Y_train = Y_train[:NUMBER_OF_TRAINING_DATA]

    #Last 10 000 pictures
    NUMBER_OF_TEST_DATA = 1000
    X_test = X_test[-NUMBER_OF_TEST_DATA:]
    Y_test = Y_test[-NUMBER_OF_TEST_DATA:]

    # Pre-process data
    #Get the inputs in range [-1,1]
    X_train, X_test = X_train / 127.5, X_test / 127.5

    X_train = X_train[:] - 1
    X_test = X_test[:] - 1
    Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

    #X_train is 54000 and X_val is 6000
    X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)


    training_data = list(zip(X_train,Y_train))
    test_data = list(zip(X_test,Y_test))
    val_data = list(zip(X_val, Y_val))

    #Hyperparameters

    INPUT_NODES = 784
    HIDDEN_LAYER_1_NODES = 59
    HIDDEN_LAYER_2_NODES = 59
    OUTPUT_NODES = 10
    sizes = [INPUT_NODES, HIDDEN_LAYER_1_NODES, HIDDEN_LAYER_2_NODES, OUTPUT_NODES]
    epochs = 10
    batch_size = 128
    check_step = X_train.shape[0]/(batch_size*100)
    learning_rate = 0.9
    momentum_parameter = 0.9

    net = Network(sizes)

    net.gradient_descent(training_data, test_data, val_data, epochs, batch_size, learning_rate, check_step, momentum_parameter)

    plt.plot(net.TRAINING_STEP, net.TRAIN_LOSS, label="Training loss")
    plt.plot(net.TRAINING_STEP, net.TEST_LOSS, label="Testing loss")
    plt.plot(net.TRAINING_STEP, net.VAL_LOSS, label="Validation loss")
    plt.legend()
    plt.ylim([0, 1])
    plt.show()

    #Plot percent classified correctly
    plt.figure(figsize=(12, 8 ))
    plt.ylim([0, 1])
    plt.ylabel("Percentage Classified Correctly ")
    plt.xlabel("Training steps")
    plt.plot(net.TRAINING_STEP, net.PERCENT_CLASSIFIED_CORRECT_TRAIN, label="Training set")
    plt.plot(net.TRAINING_STEP, net.PERCENT_CLASSIFIED_CORRECT_VAL, label="Validation set")
    plt.plot(net.TRAINING_STEP, net.PERCENT_CLASSIFIED_CORRECT_TEST, label="Test set")
    plt.legend()
    plt.show()

    #Final loss results
    print('Training loss result:', net.TRAIN_LOSS[-1])
    print('Valuation loss result:', net.VAL_LOSS[-1])
    print('Test loss result:', net.TEST_LOSS[-1])

    #Final percentage results
    print('Training percentage correct result:', net.PERCENT_CLASSIFIED_CORRECT_TRAIN[-1])
    print('Valuation percentage correct result:', net.PERCENT_CLASSIFIED_CORRECT_VAL[-1])
    print('Test percentage correct result:', net.PERCENT_CLASSIFIED_CORRECT_TEST[-1])


main()
