import matplotlib.pyplot as plt
import numpy as np


x1, y1 = np.loadtxt("train_accuracy0.001.csv",delimiter=',',unpack=True)
x2, y2 = np.loadtxt("test_accuracy0.001.csv",delimiter=',',unpack=True)
plt.plot(x1,y1, label="Training Set")
plt.plot(x2,y2, label="Testing Set")
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%) ')
plt.title('For Learning rate 0.001')
plt.legend()
plt.show()