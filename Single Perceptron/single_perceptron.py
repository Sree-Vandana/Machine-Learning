"""
Written by: Sree Vandana
Single Perceptron on mnist dataset, with different learning rate (0.001, 0.01, 0.1)
"""

import numpy as np
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix

class Perceptron:

    def __init__(self, train_data, test_data):

        # read training and test data.
        self.training_data = np.array(pd.read_csv("mnist_train.csv", header=None))
        self.testing_data = np.array(pd.read_csv("mnist_test.csv", header=None))
        # create random weights nd-array (10 X 785)
        self.weights = np.random.uniform(-0.05, 0.05, (10,785))
        # bias value X0 = 1
        self.bias = 1
        # different learning rates.
        self.learning_rate = 0.001
        #self.learning_rate = 0.01
        #self.learning_rate = 0.1


    def process(self, train, epoch):
        actual_list=[]
        predicted_list = []

        if (train == 1): # if its a training phase
            data = self.training_data
        else:            # if its a testing phase
            data = self.testing_data

        for j in range(0, data.shape[0]):  # loop to run through entire data set --> all rows Xi

            calcu_list = []
            y_list = []

            # true data value located at data[0][0] is stored
            actual_value = data[j,0].astype('int')
            actual_list.append(actual_value)

            # this is a one-hot encoding vector
            ground_truth = [0,0,0,0,0,0,0,0,0,0]
            ground_truth[actual_value] = 1

            # Normalize the data
            xi = data[j].astype('float16')/255
            xi[0] = self.bias


            for i in range(10): #run through all 10 perceptrons
                calcu_list.append(np.inner(xi, self.weights[i, :]))

                if(calcu_list[i] > 0):
                    y_list.append(1)
                else:
                    y_list.append(0)

            predicted_list.append(np.argmax(np.array((calcu_list))))  # store the maximum predicted values.

            # if its training then update weights if y!=t
            if(train == 1 and epoch >0):
                for k in range(10):
                    self.weights[k,:] = self.weights[k,:] - (self.learning_rate * (y_list[k] - ground_truth[k]) * xi)

        # calculate accuracy of training and testing data for each epoch.
        accuracy_x = (np.array(predicted_list) == actual_list).sum() / float(len(actual_list)) * 100

        # print confusion matrix on testing data.
        if(train == 0):
                print("confusion matrix for epoch ",epoch)
                print(confusion_matrix(actual_list, predicted_list))

        return accuracy_x

    def store_accuracy(self, epoch, x_accuracy, file_name):
        # store accuracies of training and testing data in a .csv file to plot it later
        with open(file_name, 'a', newline='') as file:
            wr = csv.writer(file)
            wr.writerow([epoch, x_accuracy])

train_data = "mnist_train.csv"
test_data = "mnist_test.csv"

P = Perceptron(train_data, test_data)
for epoch in range(70): # run for 70 epochs
    train_accuracy = P.process(1, epoch)
    test_accuracy = P.process(0, epoch)
    P.store_accuracy(epoch, train_accuracy, 'train_accuracy' + str(P.learning_rate) + '.csv')
    P.store_accuracy(epoch, test_accuracy, 'test_accuracy' + str(P.learning_rate) + '.csv')
