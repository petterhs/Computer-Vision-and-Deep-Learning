import numpy as np
import random

class Network:

    def __init__(self, sizes):
        #"Sizes" gives the number of nodes in each layer in the network
        # The first and last layers are inputs and outputs respectively,
        # and the rest are hidden layers
        #
        self.num_layers = len(sizes)

        self.sizes = sizes
        #self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(x,y) for x, y in zip(sizes[:-1], sizes[1:])]

        self.TRAINING_STEP = []

        self.TRAIN_LOSS = []
        self.TEST_LOSS = []
        self.VAL_LOSS = []

        self.PERCENT_CLASSIFIED_CORRECT_TRAIN = []
        self.PERCENT_CLASSIFIED_CORRECT_VAL = []
        self.PERCENT_CLASSIFIED_CORRECT_TEST = []

        self.INCREASING = []

    def feedforward(self, inputs):
        #Makes predictions for the output layer
        #Want to use sigmoid for hidden layer and
        #softmax for the output layer

        #Hidden layer
        z_hidden = np.dot(inputs, self.weights[0])
        a_hidden = sigmoid(z_hidden)
        #a_hidden = improved_sigmoid(z_hidden)  ##Uncomment for task 3##

        #Output layer
        z_output = np.dot(a_hidden, self.weights[1])
        prediction = softmax(z_output)
        return prediction, z_hidden, a_hidden

    def gradient_descent(self, training_data, test_data, val_data, epochs, mini_batch_size, learning_rate, check_step):
        #Parameters
        i = 0
        training_it = 0
        for epoch in range(epochs):
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                training_it += 1
                self.backprop(mini_batch, learning_rate)

                if i % check_step == 0:
                    self.TRAINING_STEP.append(training_it)

                    train_loss = self.cross_entropy_loss(training_data)
                    self.TRAIN_LOSS.append(train_loss)

                    test_loss = self.cross_entropy_loss(test_data)
                    self.TEST_LOSS.append(test_loss)

                    val_loss = self.cross_entropy_loss(val_data)
                    self.VAL_LOSS.append(val_loss)

                    self.PERCENT_CLASSIFIED_CORRECT_TRAIN.append(self.test_classification(training_data))
                    self.PERCENT_CLASSIFIED_CORRECT_TEST.append(self.test_classification(test_data))
                    self.PERCENT_CLASSIFIED_CORRECT_VAL.append(self.test_classification(val_data))
                i += 1
            ## Task 3 ##
            #training_data = self.train_shuffle(training_data)
			####

            #early stopping
            #early_stopping = self.early_stopping(training_it, val_loss)
            # if (early_stopping):
            #     break

            print("Epochs", epoch)


    def backprop(self, mini_batch, learning_rate):

        x, y = self.unZip(mini_batch)
        normailization_factor = x.shape[0]*y.shape[1]

        predictions, z_hidden, a_hidden = self.feedforward(x)

        delta_k = self.cost_derivative(predictions, y)

        #OUTPUT LAYER
        dw_out = np.dot(a_hidden.T, delta_k)
        dw_out /= normailization_factor
        #update
        self.weights[1] -= learning_rate*dw_out

        #HIDDEN LAYER
        delta_j = sigmoid_prime(z_hidden.T)*np.dot(dw_out, delta_k.T)
        #delta_j = improved_sigmoid_prime(z_hidden.T)*np.dot(dw_out, delta_k.T)
        dw_hidden = np.dot(delta_j, x)
        dw_hidden /= normailization_factor
        #update
        self.weights[0] -= learning_rate*dw_hidden.T


    def cross_entropy_loss(self, data):
        inputs, targets = self.unZip(data)

        #Only the first element of feedforward
        prediction = self.feedforward(inputs)[0]
        assert prediction.shape == targets.shape
        cross_entropy = -(targets * np.log(prediction))
        return cross_entropy.mean()


    def cost_derivative(self, output_activations, y):
        #Equvilent to -(t - y)
        return (output_activations-y)

    def unZip(self, data):
        X,Y = zip(*data)
        X = np.array(X)
        Y = np.array(Y)
        return X,Y

    def test_classification(self, data_set):
        X_set, Y_set = self.unZip(data_set)
        prediction = self.feedforward(X_set)[0]
        numberOfCorrect = 0
        for i in range(Y_set.shape[0]):
            if np.argmax(Y_set[i]) == np.argmax(prediction[i]) and np.amax(prediction[i]):
                #and np.amax(output[i])>=0.5):
                numberOfCorrect += 1
        return numberOfCorrect / Y_set.shape[0]


    # def early_stopping(self, training_it, val_loss):
    #     if val_loss > self.INCREASING[-1] & training_it > 20:
    #         self.INCREASING.append(val_loss)
    #         if (len(self.INCREASING) > 3):
    #
    #             #Removing the last 3 elements to get the minima (BUNNPUNKT PÅ NORSK)
    #             self.TRAIN_LOSS = self.TRAIN_LOSS[-3:]
    #             self.TEST_LOSS = self.TEST_LOSS[-3:]
    #             self.VAL_LOSS = self.VAL_LOSS[-3:]
    #             return True
    #     else:
    #         self.INCREASING = []
    #
    #         return False

    ## Task 3 funcions ##
    def train_shuffle(self, training_data):
        X, Y = self.unZip(training_data)
        dataset_size = X.shape[0]
        idx = np.arange(0, dataset_size)
        np.random.shuffle(idx)
        X  = X[idx]
        Y  = Y[idx]
        return list(zip(X,Y))

    def normal_distributed_weights(self, input_nodes, hidden_nodes, output_nodes):
        weights1 = (input_nodes, hidden_nodes)
        weights2 = (hidden_nodes, output_nodes)
        self.weights = [np.random.normal(0, 1/np.sqrt(784), weights1),
        np.random.normal(0, 1/np.sqrt(64), weights2)]


#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(a):
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)

## Task 3 ##
def improved_sigmoid(z):
    return 1.7159*np.tanh((2/3)*z)

def improved_sigmoid_prime(z):
    # term = (np.cosh(2*z/3))
    # term = np.power(term, 2)
    # return 1.14393/term
    k = (2/3)*z
    temp = np.power((np.exp(-k)+np.exp(k)),2)
    out = 4.57573/temp
    return out
