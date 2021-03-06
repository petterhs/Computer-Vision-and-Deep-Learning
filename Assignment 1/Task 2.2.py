import numpy as np
import matplotlib.pyplot as plt
import mnist
from numpy import linalg as la


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

    #Add bias
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

#Logistic Regression
def forward_pass(x, w):
    a = np.dot(x,w)
    g = 1 /(1 + np.exp(-a))
    return g

#Regularization for complexity penalty
def L2_regularization(weights, strength):
    return weights * 2 * strength

#Gradient decent for Logistic Regression
def gradient_decent(X_batch, outputs, targets, weights, learning_rate, reg_strength):
    N = X_batch.shape[0]
    assert outputs.shape == targets.shape
    dw = - X_batch * (targets - outputs)
    dw = dw.mean(axis=0)
    dw = dw.reshape(-1, 1)

    #L2_regularization
    dw_reg = L2_regularization(weights, reg_strength)
    assert dw.shape == dw_reg.shape
    dw += dw_reg

    assert dw.shape == weights.shape
    weights = weights - learning_rate * dw
    return weights

def lenght_weights(weights):
    return la.norm(weights, 'fro')

def training_loop(weights, epochs, batch_size, learning_rate, reg_strength):
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
            weights = gradient_decent(X_batch, outputs, Y_batch, weights, learning_rate, reg_strength)

            if i % check_step == 0:
                TRAINING_STEP.append(training_it)
                if reg_strength == 10000:
                    PERCENT_CLASSIFIED_CORRECT_VAL_1.append(test_classification(X_val, Y_val, weights))
                    LENGHT_WEIGHTS_1.append(lenght_weights(weights))
                elif reg_strength == 1000:
                    PERCENT_CLASSIFIED_CORRECT_VAL_2.append(test_classification(X_val, Y_val, weights))
                    LENGHT_WEIGHTS_2.append(lenght_weights(weights))
                elif reg_strength == 100:
                    PERCENT_CLASSIFIED_CORRECT_VAL_3.append(test_classification(X_val, Y_val, weights))
                    LENGHT_WEIGHTS_3.append(lenght_weights(weights))
    return weights

#Test percent classified correctly
def test_classification(X_set, Y_set, weights):
    numberOfCorrect = 0
    outputs = forward_pass(X_set, weights)
    for i in range(Y_set.shape[0]):
        if Y_set[i] == 1 and outputs[i]>=0.5:
            numberOfCorrect += 1
        elif Y_set[i] == 0 and outputs[i]<=0.5:
            numberOfCorrect += 1
    return numberOfCorrect / Y_set.shape[0]

#Plot weights as picture in 28x28 frame
def plot_weights_as_picture(weights):
    weights = weights[:784]
    image = weights.reshape(28,28)
    plt.figure(figsize=(12, 8))
    imgplot = plt.imshow(image)
    plt.colorbar()
    plt.show()


##      Run only once       ##
#mnist.init()


##      Initialize      ##
X_train, Y_train, X_test, Y_test = mnist.load()

#Split X_train set into X_train and validation set
X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)

#Resize
X_test = X_test[8000:]
Y_test = Y_test[8000:]
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
w1 = np.zeros((num_features, 1))
w2 = np.zeros((num_features, 1))
w3 = np.zeros((num_features, 1))

##      Hyperparameters     #
epochs = 2
batch_size = 10
learning_rate = 0.000001
TRAINING_STEP = []
PERCENT_CLASSIFIED_CORRECT_VAL_1 = []
PERCENT_CLASSIFIED_CORRECT_VAL_2 = []
PERCENT_CLASSIFIED_CORRECT_VAL_3 = []
LENGHT_WEIGHTS_1 = []
LENGHT_WEIGHTS_2 = []
LENGHT_WEIGHTS_3 = []


##      2.2b)       ##

#Needs to be changed in training loop aswell if altered
reg_strength_1 = 10000
reg_strength_2 = 1000
reg_strength_3 = 100

w1 = training_loop(w1, epochs, batch_size, learning_rate, reg_strength_1)
TRAINING_STEP = [] #Reset trainingstep since it is equal in each of these training loops
w2 = training_loop(w2, epochs, batch_size, learning_rate, reg_strength_2)
TRAINING_STEP = [] #Reset trainingstep since it is equal in each of these training loops
w3 = training_loop(w3, epochs, batch_size, learning_rate, reg_strength_3)

#Plot percent classified correct with 3 different regularization strengths
plt.figure(figsize=(12, 8 ))
plt.ylim([0, 1])
plt.ylabel("Percentage Classified Correctly ")
plt.xlabel("Training steps")
plt.plot(TRAINING_STEP, PERCENT_CLASSIFIED_CORRECT_VAL_1, label="Regularization strength = 10000")
plt.plot(TRAINING_STEP, PERCENT_CLASSIFIED_CORRECT_VAL_2, label="Regularization strength = 1000")
plt.plot(TRAINING_STEP, PERCENT_CLASSIFIED_CORRECT_VAL_3, label="Regularization strength = 100")
plt.legend() # Shows graph labels
plt.show()


##      2.2c)       ##

# #Plot lenght of weight vector with 3 different regularization strengths
plt.figure(figsize=(12, 8))
plt.ylim([0, 0.05])
plt.ylabel("Lenght of weights vector")
plt.xlabel("Training steps")
plt.plot(TRAINING_STEP, LENGHT_WEIGHTS_1, label="Length of weights vector with regularization strength = 10000:")
plt.plot(TRAINING_STEP, LENGHT_WEIGHTS_2, label="Length of weights vector with regularization strength = 1000:")
plt.plot(TRAINING_STEP, LENGHT_WEIGHTS_3, label="Length of weights vector with regularization strength = 100:")
plt.legend()
plt.show()


##      2.2d)       ##

#Plot weights as picture
plot_weights_as_picture(w1)
plot_weights_as_picture(w2)
plot_weights_as_picture(w3)

#Final percentage
print('Valuation percentage correct with regularization strength = 10000:', test_classification(X_val, Y_val, w1))
print('Valuation percentage correct with regularization strength = 1000:', test_classification(X_val, Y_val, w2))
print('Valuation percentage correct with regularization strength = 100:', test_classification(X_val, Y_val, w3))
