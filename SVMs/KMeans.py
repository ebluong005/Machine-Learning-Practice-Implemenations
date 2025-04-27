import numpy as np

class KMeans():
    # This function initializes the KMeans class
    def __init__(self, k=3, num_iter=1000, order=2):
        # Set a seed for easy debugging and evaluation
        np.random.seed(42)
        
        # This variable defines how many clusters to create
        # default is 3
        self.k = k

        # This variable defines how many iterations to recompute centroids
        # default is 1000
        self.num_iter = num_iter

        # This variable stores the coordinates of centroids
        self.centers = None

        # This variable defines whether it's K-Means or K-Medians
        # an order of 2 uses Euclidean distance for means
        # an order of 1 uses Manhattan distance for medians
        # default is 2
        if order == 1 or order == 2:
            self.order = order
        else:
            raise Exception("Unknown Order")

    # This function fits the model with input data (training)
    def fit(self, X):
        # m, n represent the number of rows (observations) 
        # and columns (positions in each coordinate)
        m, n = X.shape

        # self.centers are a 2d-array of 
        # (number of clusters, number of dimensions of our input data)
        self.centers = np.zeros((self.k, n))

        # self.cluster_idx represents the cluster index for each observation
        # which is a 1d-array of (number of observations)
        self.cluster_idx = np.zeros(m)

        ##### TODO 1 ######
        #
        # Task: initialize self.centers
        #
        ####################
        for i in range(n):
            lower_bound = np.percentile(X[:, i], 10)
            upper_bound = np.percentile(X[:, i], 90)
            self.centers[:, i] = np.random.uniform(lower_bound, upper_bound, self.k)
        ##### END TODO 1 #####

        for i in range(self.num_iter):
            # new_centers are a 2d-array of 
            # (number of clusters, number of dimensions of our input data)
            new_centers = np.zeros((self.k, n))

            ##### TODO 2 ######
            #
            # Task: calculate the distance and create cluster index for each observation
            #
            ####################
            distances = np.linalg.norm(X[:, np.newaxis] - self.centers, ord=self.order, axis=2)
            cluster_idx = np.argmin(distances, axis=1)
            ##### END TODO 2 #####

            ##### TODO 3 ######
            #
            # Task: calculate the coordinates of new_centers based on cluster_idx
            #
            ####################
        
            for idx in range(self.k):
                cluster_coordinates = X[cluster_idx == idx]
                if len(cluster_coordinates) > 0:
                    if self.order == 2:
                        cluster_center = np.mean(cluster_coordinates, axis=0)
                    elif self.order == 1:
                        cluster_center = np.median(cluster_coordinates, axis=0)
                    new_centers[idx, :] = cluster_center
                else:
                    new_centers[idx, :] = self.centers[idx, :]  # Keep the old center if no points are assigned
            ##### END TODO 3 #####

            ##### TODO 4 ######
            #
            # Task: determine early stop and update centers and cluster_idx
            #
            ####################
            if (cluster_idx == self.cluster_idx).all():
                print(f"Early Stopped at Iteration {i}")
                return self
            self.centers = new_centers
            self.cluster_idx = cluster_idx
            ##### END TODO 4 #####
        return self

    # This function makes predictions with input data
    # Copy-paste your code from TODO 2 and return cluster_idx
    def predict(self, X):
        ##### Predict function ######
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, ord=self.order, axis=2)
        cluster_idx = np.argmin(distances, axis=1)
        return cluster_idx