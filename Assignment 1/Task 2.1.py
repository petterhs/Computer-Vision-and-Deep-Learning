

import numpy as np
import matplotlib.pyplot as plt
import mnist


#Create a random validation set out of a % of the train data
def train_val_split(X_set, Y_set, val_percentage):
    dataset_size = X_set.shape[0]
    idx = np.arange(0, dataset_size)
    np.random.shuffle(idx)

    train_size =int(dataset_size*(1-val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]
    X_train = X_set[idx_train]
    Y_train = Y_set[idx_train]
    X_val = X_set[idx_val]
    Y_val = Y_set[idx_val]
    return X_train, Y_train, X_val, Y_val

#Filter out images with labels other than 2 or 3
#Add bias
#Reshape Y sets
def prepare_sets(X_set, Y_set):
    #Filter set
    index = 0
    filtered_X_set = X_set
    filtered_Y_set = Y_set
    for i in range(Y_set.shape[0]):
        if Y_set[i] == 2 or Y_set[i] == 3:
            filtered_X_set[index] = X_set[i]
            filtered_Y_set[index] = Y_set[i]
            index += 1
    X_set = filtered_X_set[:index]
    Y_set = filtered_Y_set[:index]

    #add bias
    X_set = np.concatenate((X_set, np.ones((X_set.shape[0], 1))), axis=1)
    #Reshape Y_set from (k,) -> (k,1)
    Y_set = Y_set.reshape(Y_set.shape[0],1)

    return X_set, Y_set

#Set target to 1 if label is 2 and 0 if label is 3
def set_target(Y_set):
    for i in range(Y_set.shape[0]):
        if Y_set[i] == 2:
            Y_set[i] = 1
        else:
            Y_set[i] = 0

#Define cross entropy loss function
def ce_loss(targets, outputs):
    assert targets.shape == outputs.shape
    error = targets*np.log(outputs) + (1-targets)*np.log(1-outputs);
    return -error.mean()


def plot_error_function(targets, outputs):
    assert targets.shape == outputs.shape
    error = targets*np.log(outputs) + (1-targets)*np.log(1-outputs);
    plt.plot(-error)
    plt.ylabel('error')
    plt.show()

#Computing output values
def forward_pass(x, w):
    a = np.dot(x,w)
    g = 1 /(1 + np.exp(-a))
    return g

def gradient_decent(X_batch, outputs, targets, weights, learning_rate):
    N = X_batch.shape[0]
    assert outputs.shape == targets.shape
    dw = - X_batch * (targets - outputs)
    dw = dw.mean(axis=0)
    dw = dw.reshape(-1, 1)
    assert dw.shape == weights.shape
    weights = weights - learning_rate * dw
    return weights

def training_loop(weights, epochs, batch_size, learning_rate):
    num_batches_per_epoch = X_train.shape[0] // batch_size
    check_step = num_batches_per_epoch // 100
    training_it = 0
    for epoch in range(epochs):
        for i in range(num_batches_per_epoch):
            training_it += 1
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]
            outputs = forward_pass(X_batch, weights)
            #print('outputs', outputs)
            weights = gradient_decent(X_batch, outputs, Y_batch, weights, learning_rate)

            if i % check_step == 0:
                TRAINING_STEP.append(training_it)

                train_out = forward_pass(X_train, weights)
                train_loss = ce_loss(Y_train, train_out)
                TRAIN_LOSS.append(train_loss)

                val_out = forward_pass(X_val, weights)
                val_loss = ce_loss(Y_val, val_out)
                VAL_LOSS.append(val_loss)

                test_out = forward_pass(X_test, weights)
                test_loss = ce_loss(Y_test, test_out)
                TEST_LOSS.append(test_loss)


                PERCENT_CLASSIFIED_CORRECT_TRAIN.append(test_classification(X_train, Y_train, weights))
                PERCENT_CLASSIFIED_CORRECT_TEST.append(test_classification(X_test, Y_test, weights))
                PERCENT_CLASSIFIED_CORRECT_VAL.append(test_classification(X_val, Y_val, weights))
    return weights

def test_classification(X_set, Y_set, weights):
    numberOfCorrect = 0
    outputs = forward_pass(X_set, weights)
    for i in range(Y_set.shape[0]):
        if Y_set[i] == 1 and outputs[i]>=0.5:
            numberOfCorrect += 1
        elif Y_set[i] == 0 and outputs[i]<=0.5:
            numberOfCorrect += 1
    return numberOfCorrect / Y_set.shape[0]



#Run only once
#mnist.init()

X_train, Y_train, X_test, Y_test = mnist.load()

#Split X_train set into X_train and validation set
X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)

#Resize
X_test = X_test[2000:]
Y_test = Y_test[2000:]
X_train = X_train[:20000]
Y_train = Y_train[:20000]

#Filter, add bias, reshape Y_set
X_train, Y_train = prepare_sets(X_train, Y_train)
X_val, Y_val = prepare_sets(X_val, Y_val)
X_test, Y_test = prepare_sets(X_test, Y_test)

#Change values of Y_set from 2 and 3 to 1 and 0
set_target(Y_train)
set_target(Y_test)
set_target(Y_val)

#Initializing weights
num_features = X_train.shape[1]
w = np.zeros((num_features, 1))

#output = forward_pass(X_train, w)
#plot_error_function(Y_train, output)

epochs = 2
batch_size = 10
learning_rate = 0.000001

TRAINING_STEP = []
TRAIN_LOSS = []
VAL_LOSS = []
TEST_LOSS = []
PERCENT_CLASSIFIED_CORRECT_TRAIN = []
PERCENT_CLASSIFIED_CORRECT_VAL = []
PERCENT_CLASSIFIED_CORRECT_TEST = []

w = training_loop(w, epochs, batch_size, learning_rate)

#Plot Cross Entropy Loss function
plt.figure(figsize=(12, 8 ))
plt.ylim([0, 1])
plt.ylabel("CE Loss")
plt.xlabel("Training steps")
plt.plot(TRAINING_STEP, TRAIN_LOSS, label="Training loss")
plt.plot(TRAINING_STEP, VAL_LOSS, label="Validation loss")
plt.plot(TRAINING_STEP, TEST_LOSS, label="Test loss")
plt.legend() # Shows graph labels
plt.show()


#Plot percent
plt.figure(figsize=(12, 8 ))
plt.ylim([0, 1])
plt.ylabel("Percentage Classified Correctly ")
plt.xlabel("Training steps")
plt.plot(TRAINING_STEP, PERCENT_CLASSIFIED_CORRECT_TRAIN, label="Training set")
plt.plot(TRAINING_STEP, PERCENT_CLASSIFIED_CORRECT_VAL, label="Validation set")
plt.plot(TRAINING_STEP, PERCENT_CLASSIFIED_CORRECT_TEST, label="Test set")
plt.legend() # Shows graph labels
plt.show()


#Final percentage
print('Training percentage correct:', test_classification(X_train, Y_train, w))
print('Valuation percentage correct:', test_classification(X_val, Y_val, w))
print('Test percentage correct:', test_classification(X_test, Y_test, w))
