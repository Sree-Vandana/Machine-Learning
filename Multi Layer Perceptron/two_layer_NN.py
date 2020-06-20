"""
written by Sree Vandana
Multi-layer Neural Network with one hidden layer
backpropagation using stochastic gradiant descent
"""

import numpy as np
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix
from scipy.special import expit

class Neural_Network:

    def __init__(self, train_data, test_data): # initializing values

        # read training and test data. (60,000 X 785) and (10,000 X 785)
        self.training_data = np.array(pd.read_csv("mnist_train.csv", header=None))
        self.testing_data = np.array(pd.read_csv("mnist_test.csv", header=None))

        # shuffle the data and choose part of it. (Exp 3) for 30,000 and 15,000
        np.random.shuffle(self.training_data)
        self.training_data = self.training_data[0:30000]

        # layers info --> hidden layer 20 units + 1 bias [20, 50, 100] (Exp 1)
        self.n = 100

        # create random weights nd-matrix (785 X 20) ; (21 X 10) for n = 20
        self.weights_xh = np.random.uniform(-0.05, 0.05, (785, self.n))
        self.weights_hk = np.random.uniform(-0.05, 0.05, (self.n+1, 10))

        # bias value X0 = 1
        self.bias = 1

        # store previous delta wt from hidden to output layer
        self.prev_wt_ho = np.zeros((self.n+1, 10))

        # store previous delta wt from input to hidden layer
        self.prev_wt_ih = np.zeros((785, self.n))

        # learning rates.
        self.learning_rate = 0.1

        # different momemtum value alpha [(n=100)--> 0.9, 0, 0.25, 0.5] (Exp 2)
        self.alpha = 0.5

        # to store activations from hidden unint
        self.activation_h= np.zeros((1, self.n+1))
        self.activation_h[0,0] = 1 #bias
        print("end of init")


    def process(self, train, epoch):
        actual_list=[]
        predicted_list = []

        if (train == 1): # if its a training phase
            data = self.training_data
        else:            # if its a testing phase
            data = self.testing_data

        for i in range(0, data.shape[0]):  # loop to run through entire data set --> all rows Xi

            # true data value located at data[i,0] is stored
            actual_value = data[i,0].astype('int')
            actual_list.append(actual_value)

            # Normalize the data
            xi = data[i].astype('float16')/255
            xi[0] = self.bias
            xi = xi.reshape(1, 785)

            # activation calculations of hidden (20 + 1) and output (10) unit
            z_h = np.dot(xi, self.weights_xh)
            act_h = expit(z_h)      # sigma fn
            self.activation_h[0,1:] =act_h

            z_k = np.dot(self.activation_h, self.weights_hk)
            act_k = expit(z_k)      # sigma fn

            # store max of output units activations
            predicted_list.append(np.argmax(np.array((act_k))))

            # *** backpropagation using stochastic gradiant descent ***

            if (epoch > 0 and train == 1):

                # error calculation
                tk = np.zeros((1, 10)) + 0.1
                tk[0, actual_value] = 0.9

                # error at output
                error_ok = act_k * (1 - act_k) * (tk - act_k)
                #print("error_ok= ",error_ok.shape)

                # error at hidden units
                error_oh = act_h * (1 - act_h) * (np.dot(error_ok, self.weights_hk[1:, :].T))
                #print("error_oj shap--> ",error_oj.shape)

                # update weights (h --> O and x --> h)

                delta_weight_ho = (self.learning_rate * error_ok * self.activation_h.T) + (self.alpha * self.prev_wt_ho)
                self.prev_wt_ho = delta_weight_ho
                self.weights_hk = self.weights_hk + delta_weight_ho

                delta_weight_ih = (self.learning_rate * error_oh * xi.T) + (self.alpha * self.prev_wt_ih)
                self.prev_wt_ih = delta_weight_ih
                self.weights_xh = self.weights_xh + delta_weight_ih

        # calculate accuracy of training and testing data for each epoch.
        accuracy_x = (np.array(predicted_list) == actual_list).sum() / float(len(actual_list)) * 100

        # print confusion matrix on testing data.
        if(epoch > 0):
                if(train == 1):
                    print("for testing")
                else:
                    print("for training")
                print("confusion matrix for epoch ",epoch)
                print(confusion_matrix(actual_list, predicted_list))

        return accuracy_x


    def store_accuracy(self, epoch, x_accuracy, file_name):
        # store accuracies of training and testing data in a .csv file to plot it later using (tow_layer_NN_plot.py)
        with open(file_name, 'a', newline='') as file:
            wr = csv.writer(file)
            wr.writerow([epoch, x_accuracy])

# File Names
train_data = "mnist_train.csv"
test_data = "mnist_test.csv"

# calls init to initialize values before sending testing set.
NN = Neural_Network(train_data, test_data)
for epoch in range(50): # run for epochs
    train_accuracy = NN.process(1, epoch)
    test_accuracy = NN.process(0, epoch)
    NN.store_accuracy(epoch, train_accuracy, 'train_acc' + str(NN.learning_rate) + '_' +str(NN.alpha) + '_' +str(NN.n) + '_' +'30k'+ '.csv')
    NN.store_accuracy(epoch, test_accuracy, 'test_acc' + str(NN.learning_rate) + '_' + str(NN.alpha)+ '_' +str(NN.n)+ '_' +'30k' + '.csv')







