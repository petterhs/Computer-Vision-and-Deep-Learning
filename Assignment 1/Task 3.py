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


def prepare_sets(X_set, Y_set, num_classes):
    #add bias
    X_set = np.concatenate((X_set, np.ones((X_set.shape[0], 1))), axis=1)
    #one_hot_encoding of Y_set
    Y_set = np.eye(num_classes)[Y_set]
    return X_set, Y_set

#Computing output values
def forward_pass(x, w):
    a = np.dot(x,w)
    g = 1 /(1 + np.exp(-a))
    return g

def ce_loss_softmax(targets, outputs):
    assert targets.shape == outputs.shape
    error = targets*np.log(outputs)
    return -error.mean()
def L2_regularization(weights, strength):
    return weights * 2 * strength
def gradient_decent(X_batch, outputs, targets, weights, learning_rate):
    N = X_batch.shape[0]
    assert outputs.shape == targets.shape
    # term = (targets - outputs).T
    dw = -np.dot((targets - outputs).T, X_batch).T

    assert dw.shape == weights.shape
    weights = weights - learning_rate * dw
    return weights

def training_loop(weights, epochs, batch_size, learning_rate):
    num_batches_per_epoch = X_train.shape[0] // batch_size
    check_step = num_batches_per_epoch // 10
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
                train_loss = ce_loss_softmax(Y_train, train_out)
                TRAIN_LOSS.append(train_loss)

                val_out = forward_pass(X_val, weights)
                val_loss = ce_loss_softmax(Y_val, val_out)
                VAL_LOSS.append(val_loss)

                test_out = forward_pass(X_test, weights)
                test_loss = ce_loss_softmax(Y_test, test_out)
                TEST_LOSS.append(test_loss)


                PERCENT_CLASSIFIED_CORRECT_TRAIN.append(test_classification(X_train, Y_train, weights))
                PERCENT_CLASSIFIED_CORRECT_TEST.append(test_classification(X_test, Y_test, weights))
                PERCENT_CLASSIFIED_CORRECT_VAL.append(test_classification(X_val, Y_val, weights))
    return weights

def test_classification(X_set, Y_set, weights):
    numberOfCorrect = 0
    outputs = forward_pass(X_set, weights)
    for i in range(Y_set.shape[0]):
        if (np.argmax(Y_set[i]) == np.argmax(outputs[i])):
            numberOfCorrect += 1
    return numberOfCorrect / Y_set.shape[0]




##Test set target


##Run only once
#mnist.init()

X_train, Y_train, X_test, Y_test = mnist.load()

#Split X_train set into X_train and validation set
X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)

#Resize
X_test = X_test[8000:]
Y_test = Y_test[8000:]
X_train = X_train[:1000]
Y_train = Y_train[:1000]

classes = 10
X_train, Y_train = prepare_sets(X_train, Y_train, classes)
X_val, Y_val = prepare_sets(X_val, Y_val, classes)
X_test, Y_test = prepare_sets(X_test, Y_test, classes)



# ##main
#
# #Initializing weights
num_features = X_train.shape[1]
w = np.zeros((num_features, 10))
print('weigth shape', w.shape)

output = forward_pass(X_train, w)
#Hyperparameters
epochs = 2
batch_size = 10
learning_rate = 0.000001
reg_strength_1 = 1000

TRAINING_STEP = []
TRAIN_LOSS = []
VAL_LOSS = []
TEST_LOSS = []
PERCENT_CLASSIFIED_CORRECT_TRAIN = []
PERCENT_CLASSIFIED_CORRECT_VAL = []
PERCENT_CLASSIFIED_CORRECT_TEST = []

#Run training loop
w = training_loop(w, epochs, batch_size, learning_rate)

#Plot Cross Entropy Loss function
plt.figure(figsize=(12, 8 ))
plt.ylim([0, 5000])
plt.ylabel("CE Loss")
plt.xlabel("Training steps")
plt.plot(TRAINING_STEP, TRAIN_LOSS, label="Training loss")
plt.plot(TRAINING_STEP, VAL_LOSS, label="Validation loss")
plt.plot(TRAINING_STEP, TEST_LOSS, label="Test loss")
plt.legend()
plt.show()


#Plot percent classified correctly
plt.figure(figsize=(12, 8 ))
plt.ylim([0, 1])
plt.ylabel("Percentage Classified Correctly ")
plt.xlabel("Training steps")
plt.plot(TRAINING_STEP, PERCENT_CLASSIFIED_CORRECT_TRAIN, label="Training set")
plt.plot(TRAINING_STEP, PERCENT_CLASSIFIED_CORRECT_VAL, label="Validation set")
plt.plot(TRAINING_STEP, PERCENT_CLASSIFIED_CORRECT_TEST, label="Test set")
plt.legend()
plt.show()


#Final loss results
print('Training loss result:', TRAIN_LOSS[-1])
print('Valuation loss result:', VAL_LOSS[-1])
print('Test loss result:', TEST_LOSS[-1])

#Final percentage results
print('Training percentage correct result:', test_classification(X_train, Y_train, w))
print('Valuation percentage correct result:', test_classification(X_val, Y_val, w))
print('Test percentage correct result:', test_classification(X_test, Y_test, w))
