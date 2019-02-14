
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
        self.velocities = [0 for x, y in zip(sizes[:-1], sizes[1:])]
        self.TRAINING_STEP = []

        self.TRAIN_LOSS = []
        self.TEST_LOSS = []
        self.VAL_LOSS = []

        self.PERCENT_CLASSIFIED_CORRECT_TRAIN = []
        self.PERCENT_CLASSIFIED_CORRECT_VAL = []
        self.PERCENT_CLASSIFIED_CORRECT_TEST = []

        self.INCREASING = [0]
    def feedforward(self, inputs):
        #Makes predictions for the output layer
        #Want to use sigmoid for hidden layer and
        #softmax for the output layer

        #Hidden layer


        z_hidden_2 = np.dot(inputs, self.weights[0])
        a_hidden_2 = sigmoid(z_hidden_2)

        z_hidden = np.dot(a_hidden_2, self.weights[1])
        a_hidden = sigmoid(z_hidden)


        #Output layer
        z_output = np.dot(a_hidden, self.weights[2])

        prediction = softmax(z_output)

        return prediction, z_hidden_2, z_hidden, a_hidden_2, a_hidden

    def gradient_descent(self, training_data, test_data, val_data, epochs, mini_batch_size, learning_rate, check_step, momentum_parameter):
        #Parameters
        i = 0
        training_it = 0
        for epoch in range(epochs):
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                training_it += 1
                self.backprop(mini_batch, learning_rate, momentum_parameter)

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
            training_data = self.train_shuffle(training_data)
			####

            #early stopping
            early_stopping = self.early_stopping(training_it, val_loss)
            if (early_stopping):
                break

            print("Epochs", epoch)


    def backprop(self, mini_batch, learning_rate, momentum_parameter):

        x, y = self.unZip(mini_batch)
        normailization_factor = x.shape[0]*y.shape[1]

        predictions, z_hidden_2, z_hidden, a_hidden_2, a_hidden = self.feedforward(x)

        delta_k = self.cost_derivative(predictions, y)

        #OUTPUT LAYER
        dw_out = np.dot(a_hidden.T, delta_k)
        dw_out /= normailization_factor

        #HIDDEN LAYER
        delta_j = sigmoid_prime(z_hidden.T)*np.dot(self.weights[2], delta_k.T)
        #delta_j = improved_sigmoid_prime(z_hidden.T)*np.dot(self.weights[2], delta_k.T) #Use this in task 3
        #delta_j = ReLU_prime(z_hidden.T)*np.dot(self.weights[2], delta_k.T)            #Use this in task 5

        dw_hidden = np.dot(delta_j, a_hidden_2)
        dw_hidden /= normailization_factor


        #HIDDEN LAYER 2
        delta_j_2 = sigmoid_prime(z_hidden_2.T)*np.dot(self.weights[1], delta_j)
        #delta_j = improved_sigmoid_prime(z_hidden_2.T)*np.dot(self.weights[1], delta_k.T) #Use this in task 3
        #delta_j = ReLU_prime(z_hidden_2.T)*np.dot(self.weights[1], delta_k.T)            #Use this in task 5

        dw_hidden_2 = np.dot(delta_j_2, x)
        dw_hidden_2 /= normailization_factor


        #3d - Nesterov momentum
        self.velocities[0] = momentum_parameter *self.velocities[0] - learning_rate*dw_hidden_2.T
        self.velocities[1] = momentum_parameter *self.velocities[1] - learning_rate*dw_hidden.T
        self.velocities[2] = momentum_parameter *self.velocities[2] - learning_rate*dw_out

        self.weights[2] += self.velocities[2]
        self.weights[1] += self.velocities[1]
        self.weights[0] += self.velocities[0]



    def cross_entropy_loss(self, data_set):
        inputs, targets = self.unZip(data_set)

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

    def early_stopping(self, training_it, val_loss):
        if  val_loss > self.INCREASING[-1] & training_it > 20:
            self.INCREASING.append(val_loss)
            if (len(self.INCREASING) > 4):

                #Removing the last 3 elements to get the minima (BUNNPUNKT PÅ NORSK)
                self.TRAIN_LOSS = self.TRAIN_LOSS[-3:]
                self.TEST_LOSS = self.TEST_LOSS[-3:]
                self.VAL_LOSS = self.VAL_LOSS[-3:]
                return True
        else:
            self.INCREASING = [0]
            return False

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
    k = (2/3)*z
    temp = np.power((np.exp(-k)+np.exp(k)),2)
    out = 4.57573/temp
    return out

def ReLU(z):
    return np.maximum(z, 0)

def ReLU_prime(z):
    #Derivative of ReLU
    return (z > 0) * 1.0
