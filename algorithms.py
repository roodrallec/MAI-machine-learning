import scipy as sp
from scipy.io import arff
from cStringIO import StringIO
import pandas
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import sklearn.metrics.cluster as sk_cluster_m
import sklearn.metrics as skmetrics

def squares_dist(x):
    # squares_dist(x=ndarray):
    # x: matrix N,M. N rows of data variables.
    # .  M/2 columns are data features values, M/2 columns are cluster centroid coord.
    # returns square distances

    return (sp.spatial.distance.pdist([x[:x.shape[0] / 2], x[x.shape[0] / 2:]],
                                      'euclidean')) ** 2


def cost_function(data_n, clusters_n, log_level=1):
    # cost_function(data_n=ndarray,clusters_n:array)
    # data_n: clustered data
    # clusters_n: clusters association in the data (size: rows of data_n, 1)
    # returns the cost function value

    # Find the centroids of each cluster
    mus = np.array([data_n[np.where(clusters_n == k)].mean(axis=0) for k in
                    range(len(np.unique(clusters_n)))])

    # vector of mu feature values of the associated cluster for each data variable
    mus_complete = np.empty([clusters_n.shape[0], 4])

    for k in range(len(np.unique(clusters_n))):
        mus_complete[np.where(clusters_n == k)] = mus[k]

    # calculate cost function
    cost_f = sum(np.apply_along_axis(squares_dist, axis=1,
                                     arr=np.concatenate((data_n, mus_complete),
                                                        axis=1)))
    cost_f = cost_f / data_n.shape[0]

    if log_level:
        print "Cost function kmeans split:", cost_f
    return cost_f


def select_split_cluster(clusters, criteria="larger", log_level=1):
    # select_split_cluster (clusters, criteria)
    # clusters: vector of data cluster association (1 column)
    # criteria= "larger". (more option tbi)
    # returns de number of the selected cluster.

    selected_key_c = 0
    number_of_x = []

    if criteria == "larger":
        for i in np.nditer(np.unique(clusters)):
            number_of_x.append([len(clusters[np.where(clusters == i)]), i])

        selected_key_c = number_of_x[number_of_x.index(max(number_of_x))][1]

        if log_level:
            print "Number of x in each cluster:", number_of_x

    return selected_key_c


def Bk_means(X, K, k_means_iter=3, log_level=1):
    # Bk_means(X=ndarray, K=Int, k_means_iter=3)
    # X: data to cluster
    # K: number of clusters
    # k_means_iter: number of iteretations on the kmeans call. # of split cluster pairs.
    # log_level: 0 : no messages, 1 print messages

    # Initialize cluster  assigment with all data
    clusters = np.zeros((X.shape[0], 1))

    # Set initial number of cluster to 1 and iterate until number of clusters=K

    for k in range(1, K):
        if log_level:
            print "*********** NEW ITERATION ************* ", k
        similarity = []
        potential_new_clusters = {}

        if log_level:
            print "*********select cluster to split******"

        larger_cluster_index = select_split_cluster(clusters, "larger",
                                                    log_level)  # options: larger, heterogeny,
        if log_level:
            print "Selected cluster: ", larger_cluster_index

        kmeans_data = X[np.where(clusters == larger_cluster_index), :]
        kmeans_data = kmeans_data[0]

        if log_level:
            print "*********Generate 2 clusters with Kmeans ******"
            print "*********Best of ", k_means_iter, " results ******"

        for i in range(0, k_means_iter):
            # if k_means_iter >1 then we select best k_means split with similarity
            # potential_new_clusters[i] = KMeans(2, "random",1).fit_predict(kmeans_data)
            potential_new_clusters[i] = KMeans(2).fit_predict(kmeans_data)
            similarity.append(
                cost_function(kmeans_data, potential_new_clusters[i],
                              log_level))

        # Select division based on similarity (min value max similarity)
        selected_division = potential_new_clusters[
            similarity.index(min(similarity))]

        if log_level:
            print "Selected case: ", similarity.index(min(similarity))

        new_clusters = selected_division
        new_clusters[np.where(selected_division == 1)] = k
        new_clusters[np.where(selected_division == 0)] = larger_cluster_index

        clusters[np.where(clusters[:] == larger_cluster_index)] = new_clusters

    if log_level:
        print "****** END OF BKmeans *********\n\n\n"
    return clusters.flatten()



# KMEANS

## Function Utils
def rand_centroids(K, X):
    # rand_centroids(K=Int, X=Float_array):
    # Return a numpy array of size K with each element
    # being a normally random distributed var with mu and sigma calculated
    # from the mean and std of the data X
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    clusters = [np.random.normal(mean, std) for n in range(K)]
    return np.array(clusters)


def euc_distance(X, Y):
    # euc_distance(X=Float_array, Y=Float_array):
    # Returns an array of euclidean distances,
    # for the square root of the sum of the square of the differences
    # of array X and array Y
    diff = X - Y[:, np.newaxis]
    squared_diff = diff ** 2
    sum_squared_diff = squared_diff.sum(axis=2)
    return np.sqrt(sum_squared_diff)


def compute_clusters(K, C, X):
    # compute_clusters(K=Int, C=Float_array, X=Float_array)
    # Compute the clusters for cluster size K, clusters C and data X
    # where a new cluster is calculated as the mean of the data points
    # which share a common nearest cluster. Repeats until the sum of
    # the euc distances between clusters and points does not change,
    # then returns the clusters
    D = euc_distance(X, C)
    CC = np.argmin(D, axis=0)
    C = np.array([new_cluster(k, X, CC) for k in range(K)])
    D2 = euc_distance(X, C)
    if (D.sum() == D2.sum()):
        return C, np.argmin(D2, axis=0)
    else:
        return compute_clusters(K, C, X)


def new_cluster(k, X, CC):
    # Returns a new cluster based on the mean of the points associated with it
    # if no points associated with it, generates a new one
    x = X[CC == k]
    if (len(x) > 0):
        return x.mean(axis=0)
    else:
        return rand_centroids(1, X)[0]


def k_means(K, X):
    # k_means(K=Int, X=Float_array)
    # K-means for clust size K on dataset X using random initialised centroids
    # returns final clusters and predicted labels
    C = rand_centroids(K, X)
    return compute_clusters(K, C, X)
