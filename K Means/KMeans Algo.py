"""
K-Means Clustering Algorithm
Sree Vandana Nadipalli
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

class KMeans:

    def __init__(self, k, data, max_iterations, t):
        self.k = k
        self.data = data
        self.t = t
        self.max_iterations = max_iterations
        # store clusered data index's
        self.clusters = [[] for _ in range(self.k)]
        # the center of each 'K' clusters
        self.centroids = []

    def initial_seeds(self):
        # randomly choose centers of each cluster
        initial_centroids = np.random.permutation(self.data.shape[0])[:k]
        self.centroids = self.data[initial_centroids]

        return self.centroids

    def expectation_maximization(self):

        self.n_samples, self.n_features = self.data.shape

        for i in range(self.max_iterations):

            # STEP 1
            create_clusters = [[] for _ in range(self.k)]

            # find closest distance between each point in data(x,y) and mean/centroid (x,y)
            for index, sample in enumerate(self.data):
                distances = [np.sqrt(np.sum((sample - c_point) ** 2)) for c_point in self.centroids]
                closest_index = np.argmin(distances)
                # add the data point to the closest cluster
                create_clusters[int(closest_index)].append(index)
                self.clusters = create_clusters

            # STEP 2
            # find new centroids of the new cluster formed.. and save old cluster
            old_centroids = self.centroids
            new_centroids = np.zeros((self.k, self.n_features))
            for cluster_index, cluster in enumerate(self.clusters):
                cluster_mean = np.mean(self.data[cluster], axis=0)
                new_centroids[cluster_index] = cluster_mean

            self.centroids = new_centroids

            # check if clusters are same or not; if same break and stop
            if self._is_converged(old_centroids, self.centroids):
                print("centroids converged, breaking the loop")
                break

    def _is_converged(self, old_centroids, centroids):
        # distances between each old and new centroids, for all centroids
        distances = [np.sqrt(np.sum((old_centroids[i] - centroids[i]) ** 2)) for i in range(self.k)]
        return sum(distances) == self.t

    def plot_clusters(self):
        global r_val, k
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
k = 8
print("k = ", k)
for r in range(10):

    KM = KMeans(k, data=np_data, max_iterations=100, t = 0)
    centroids = KM.initial_seeds()
    KM.expectation_maximization()

    # calculating sum-square-error
    error = 0
    for i, index in enumerate(KM.clusters):
        for j in range(np_data[index].shape[0]):    # for all data points within this cluster
            error = error + (np.sum(np_data[index][j] - np.array(KM.centroids)[i])**2)

    sum_square_error.append(error)
    print("r = ", r+1,"  sum-square-error = ", error)

    KM.plot_clusters()

print("errors = ", sum_square_error)
x = np.argmin(sum_square_error)
print("min error when r =",x+1, "\nerror = ", sum_square_error[x] )