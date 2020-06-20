"""
Fuzzy c means Algorithm (Unsupervised Learning)
Sree Vandana Nadipalli
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import operator

class fuzzycmeans:

    def __init__(self, c, m, data, max_iterations, t):
        self.c = c
        self.m = m
        self.data = data    #np array
        self.t = t
        self.max_iterations = max_iterations
        self.n_samples, self.n_features = self.data.shape  # 1500,2
        # store clusered data index's
        self.clusters = [[] for _ in range(self.c)] # list of lists
        # the center of each 'c' clusters
        self.centroids = [] #list
        self.old_centroids = []
        self.mem_matrix =[]

    def initial_seeds(self):
        # randomly choose centers of each cluster
        initial_centroids = np.random.permutation(self.data.shape[0])[:self.c]
        print("initial centroid indexes = ", initial_centroids)
        self.centroids = self.data[initial_centroids]

        return self.centroids

    def initial_membership_matrix(self):
        membership_mat = list()
        for i in range(self.n_samples):
            random_num_list = [random.random() for i in range(self.c)]
            summation = sum(random_num_list)
            temp_list = [x / summation for x in random_num_list]
            membership_mat.append(temp_list)
        #print("membership_mat", membership_mat)
        return membership_mat

    def calculateClusterCenter(self):
        # single list of all weights..
        cluster_mem_val = list(zip(*self.mem_matrix))
        #print("self.mem_matrix",cluster_mem_val)
        cluster_centers = list()
        # for every cluster find a center
        for j in range(self.c):
            w = list(cluster_mem_val[j])
            w_raised = [wj ** self.m for wj in w]
            denominator = sum(w_raised)
            s = [0, 0]
            # for all data sample points.. and coressponding weights w.r.t ith data point in jth cluster
            for i in range(self.n_samples):
                s = s+ (self.mem_matrix[i][j]**self.m) * self.data[i]

            center = [z / denominator for z in s]
            cluster_centers.append(center)

        return cluster_centers

    def updateMembershipValue(self):
        np.seterr(divide='ignore')
        power = float(2 / (m - 1))
        for i in range(self.n_samples):    # for all data points
            x = list(self.data[i])
            distances = [np.linalg.norm(list(map(operator.sub, x, self.centroids[j]))) for j in range(self.c)]
            #print("distances-->", distances)
            for j in range(self.c):
                den = sum([math.pow(float(distances[j] / distances[c]), power) for c in range(self.c)])
                #print("den = ", den)
                self.mem_matrix[i][j] = float(1 / den)
        return self.mem_matrix

    def getClusters(self):
        cluster_labels = [[] for _ in range(self.c)]  # list of lists
        cluster_num = 0
        for i in range(self.n_samples):
            max_weight = 0
            for j in range(self.c):
                if (np.array(self.mem_matrix[i][j]) > max_weight):
                    max_weight = self.mem_matrix[i][j]
                    cluster_num = j  # cluster index which the data point belongs to.

            cluster_labels[cluster_num].append(i)

        #print("size of clusterlabels=", len(cluster_labels))
        return cluster_labels

    def _is_converged(self, old_centroids, centroids):
        # distances between each old and new centroids, for all centroids
        distances = [np.sqrt(np.sum(np.array((old_centroids[i] - centroids[i]) ** 2))) for i in range(self.c)]
        return sum(distances) <= self.t

    def expectation_maximization(self):

        # initializations
        self.centroids = self.initial_seeds()
        self.mem_matrix = self.initial_membership_matrix()

        for i in range(self.max_iterations):
            # push data points index into clusters
            self.clusters = self.getClusters()

            if i> 0:
                self.old_centroids = self.centroids

            # calculate centroids
            self.centroids = self.calculateClusterCenter()

            #update membership values
            weights_updated = self.updateMembershipValue()

            # check convergence condition
            if i > 0:
                if self._is_converged(np.array(self.old_centroids), np.array(self.centroids)):
                    print("centroids converged, breaking the loop")
                    break


    def plot_clusters(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, index in enumerate(self.clusters):
            point = self.data[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="*", color='black', linewidth=2)

        plt.show()

# Read data and store as matrix
data = pd.read_csv("cluster_dataset.csv", header=None, dtype=float)
np_data = data.to_numpy()

sum_square_error = []

# Number of Clusters
c = 8
print("c = ", c)

for r in range(10):

    # Fuzzyfier
    m = 1.10

    # for each 'r' random weights will be assigned for each data points in _init_ function
    FCM = fuzzycmeans(c, m, data=np_data, max_iterations=50, t = 0.0001)

    # based on weights push each data points into respective clusters; and update weights and assign new clusters to data points
    FCM.expectation_maximization()

    error = 0
    for i in range(FCM.n_samples):  # for all data points
        x = list(FCM.data[i])
        one =  [FCM.mem_matrix[i][j]** m for j in range(c)]
        two = [(np.linalg.norm(list(map(operator.sub, x, FCM.centroids[j])))**2) for j in range(c)]
        error = error + sum(x * y for x, y in zip(one, two))

    sum_square_error.append(error)
    print("r = ", r+1,"sum-square-error = ", error," \n" )

    FCM.plot_clusters()

print("errors = ", sum_square_error)
x = np.argmin(sum_square_error)
print("min error when r =",x+1, "\nerror = ", sum_square_error[x])