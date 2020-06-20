# Gaussian Naïve Bayes Classification
"""
ML Programming Assignment 2
Sree Vandana Nadipalli
"""

import numpy as np
import pandas as pd
import statistics
from sklearn.metrics import confusion_matrix


# Read data and store as matrix
data = pd.read_csv("spambase.data", header=None, dtype=float);
np_data = data.to_numpy();

# separate the data into spam and non_spam data
# 1 to 1813 belong to class 1 (spam)
spam_data = np_data[:1831,:]

# 1814 to 4601 (2787) belong to class 0 (Not Spam)
not_spam_data = np_data[1814:,:]

#shuffle the data
np.random.shuffle(spam_data)
np.random.shuffle(not_spam_data)

# splitting into 40% (spam) and 60% (not_spam), for each training and testing data set
train_spam = spam_data[:930,:]
train_not_spam = not_spam_data[:1390,:]
train_data = np.concatenate((train_spam,train_not_spam),axis=0)
print("training data size ",train_data.shape)

# shuffle again
np.random.shuffle(spam_data)
np.random.shuffle(not_spam_data)

test_spam = spam_data[:930,:]
test_not_spam = not_spam_data[:1390,:]
test_data = np.concatenate((test_spam,test_not_spam),axis=0)
target_class = test_data[:,57]
print("testing data size ", test_data.shape)

# Computing Prior Probability of testing and training dataset
# For Training Dataset
count_of_spam = count_of_notspam = 0

for i in range(0, train_data.shape[0]):
	if(train_data[i,57] == 1):
		count_of_spam += 1
	else:
		count_of_notspam += 1

prior_train_spam_p = count_of_spam / len(train_data);
print("Prior training Probability for Spam ", prior_train_spam_p)

prior_train_notspam_p = count_of_notspam / len(train_data);
print("Prior training Probability for Not Spam",prior_train_notspam_p)

#computing feature(57) wise mean and std dev in training data set
spam_mean = []
notspam_mean = []

spam_std_dev = []
notspam_std_dev = []

for i in range(0,train_data.shape[1]):  # for 57 features (1,58) column
    spam_array = []
    notspam_array = []

    for j in range(train_data.shape[0]): #row
        if (train_data[j][57] == 1):
            spam_array.append(train_data[j][i])
        else:
            notspam_array.append(train_data[j][i])

    spam_mean.append(statistics.mean(spam_array))
    notspam_mean.append(statistics.mean(notspam_array))

    spam_std_dev.append(statistics.stdev(spam_array))
    notspam_std_dev.append(statistics.stdev(notspam_array))

# assigning minimal standard deviation (0.0001) to avoid a divide-by-zero error in Gaussian Naïve Bayes.
for i in range(len(spam_std_dev)):
	if (spam_std_dev[i] == 0):
		spam_std_dev[i] = 0.0001

	if (notspam_std_dev[i] == 0):
		notspam_std_dev[i] = 0.0001

# Running Naïve Bayes on the test data.
classification_result = []

# classify the test datatset
def Naive_Bayes(x, mean, std):
	np.seterr(divide='ignore')
	part1 = float(1 / (np.sqrt(2 * np.pi * std)))
	part2 = float(np.exp(-1 * ((x - mean)**2)/(2 * (float(std)**2))))
	result = part1 * part2
	return result

for i in range(test_data.shape[0]):
	# log(prior_probabilities)
	l1 = np.log(prior_train_spam_p)
	l2 = np.log(prior_train_notspam_p)

	for j in range(0,57):
		x = test_data[i][j]
		l1 += np.log(Naive_Bayes(x, spam_mean[j], spam_std_dev[j]))
		l2 += np.log(Naive_Bayes(x, notspam_mean[j], notspam_std_dev[j]))
	classification = np.argmax([l2, l1])
	classification_result.append(classification)

# confusion matrix
confusion_matrix = confusion_matrix(target_class, classification_result)
print("Confusion matrix\n",confusion_matrix)

# calculating accuracy, precision and recall

true_positive = confusion_matrix[0,0]
true_negative = confusion_matrix[1,1]
false_positive = confusion_matrix[0,1]
false_negative = confusion_matrix[1,0]

accuracy = float((true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive))
precision = float((true_positive) / (true_positive + false_positive))
recall = float((true_positive) / (true_positive + false_negative))

print("Accuracy in Percentage- ",accuracy*100)
print("Precision in Percentage- ",precision*100)
print("Recall in Percentage- ",recall*100)


