"""
need to run after two_layer_NN.py
plots the graph of Accuracies of training and testing data
"""

import numpy as np
import matplotlib.pyplot as plt

# read data points from file
x1, y1 = np.loadtxt("train_acc0.1_0.5_100.csv",delimiter=',',unpack=True)
x2, y2 = np.loadtxt("test_acc0.1_0.5_100.csv",delimiter=',',unpack=True)
plt.title('For Learning rate 0.1 and n=100 (momentum = 0.5)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.plot(x1,y1, label="Training Set")
plt.plot(x2,y2, label="Testing Set")
plt.legend()
plt.show()