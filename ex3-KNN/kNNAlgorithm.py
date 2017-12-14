# coding=utf-8
"""
    Write a Python function for classifying, using a KNN algorithm, each instance from the TestMatrix using the
    TrainMatrix to a classifier called kNNAlgorithm(…). You decide the parameters for this classifier.
    Justify your implementation and add all the references you have considered for your decisions.
    K IS A HYPER PARAMETER.
"""
import numpy as np
import scipy as sp
np.set_printoptions(linewidth=120)
np.random.seed(0)


class kNNAlgorithm(object):
    def __init__(self, k, metric="euclidean", p=4, policy="voting", weights=None, selection=None):
        self.k = k
        self._select_metric_function(metric, p)
        self._select_policy_function(policy)
        self.weights = weights
        self.selection = selection
        self.p = p

    def _select_policy_function(self, policy):
        """
        Selector function for setting the policy function.
        :param policy: name of the policy, one of "voting", "nearest"
        """
        if policy == "voting":
            self.policy_fun = self._policy_voting
        else:
            self.policy_fun = self._policy_nearest

    def _select_metric_function(self, metric, q):
        """
        Selector function for setting the metric function
        :param metric: name of the metric, one of "hamming", "cosine", "other",
        "euclidean"
        """
        if metric == "hamming":
            self.metric_fun = self._metric_hamming
        elif metric == "cosine":
            self.metric_fun = self._metric_cosine
        elif metric == "correlation":
            self.metric_fun = self._metric_correlation
        elif metric == "minkowski":
            self.metric_fun = self._metric_minkowski
        else:
            self.metric_fun = self._metric_euclidean

    def _policy_voting(self, dist_idx_mat):
        """
        Implement prediction policy by taking k nearest neighbours and
        return the most common class
        :param dist_idx_mat: distance matrix with sort indexes (N_samples x N_train)
        :return: prediction  vector (N_samples x 1)
        """

        pred = np.zeros((dist_idx_mat.shape[0],))
        res = self.train_y[dist_idx_mat[:, np.arange(self.k)]]
        for j in np.arange(res.shape[0]):
            pred[j] = self._mode(res[j, :])
        return pred

    def _mode(self, x):
        """
        Custom mode function. Finds the most frequent value in a vector efficiently.
        If more than one mode exists, it selects one value randomly.
        :param x: vector of elements
        :return: most frequent value
        """
        xs = np.sort(x)
        start = np.hstack([np.array([True]), xs[:-1] != xs[1:]])
        start = np.where(start)[0]
        freq = np.zeros(xs.size)
        freq[start] = np.hstack([np.diff(start), xs.size - start[-1]])
        idx_maxs = np.where(freq == np.max(freq))[0]
        return xs[np.random.choice(idx_maxs, 1)]

    def _policy_nearest(self, dist_idx_mat):
        """
        Implement prediction policy by taking the nearest neighbour
        :param dist_idx_mat: distance matrix with sort indexes (N_samples x N_train)
        :return: prediction  vector (N_samples x 1)
        """
        return self.train_y[dist_idx_mat[:, 0]]

    def _metric_euclidean(self, X, X_samples):
        """
        Implements euclidean distance, computes the distance from each X_sample to
        all X individuals.
        X_samples row. Assuming numeric values.
        :param X: Matrix (N_train x M) (Individuals x Features)
        :param X_samples: Matrix (N_samples x M) (Individuals x Features)
        :return: Matrix of euclidean distances (N_samples x N_train)
        """
        dist_mat = np.zeros((X_samples.shape[0], X.shape[0]))
        for j in np.arange(X_samples.shape[0]):
            if self.weights is None:
                diff = X - X_samples[j, :]
            else:
                diff = X * self.weights - X_samples[j, :] * self.weights
            dist_vec = np.sqrt(np.nansum(np.power(diff, 2), axis=1))
            dist_mat[j, :] = dist_vec

        return dist_mat

    def _metric_hamming(self, X, X_samples):
        """
        Implements Hamming distance metric as the average of a boolean vector
        resulted from comparing two vectors element wise. Reference:
        https://github.com/scipy/scipy/blob/v0.19.1/scipy/spatial/distance.py#L547
        :param X: Matrix (N_train x M) (Individuals x Features)
        :param X_samples: Matrix (N_samples x M) (Individuals x Features)
        :return: Matrix of Hamming distances (N_samples x N_train)
        """
        dist_mat = np.zeros((X_samples.shape[0], X.shape[0]))
        for j in np.arange(X_samples.shape[0]):
            if self.weights is None:
                dist_mat[j, :] = np.mean(X != X_samples[j, :], axis=1)
            else:
                dist_mat[j, :] = np.mean((X * self.weights) != (X_samples[j, :] * self.weights), axis=1)

        return dist_mat

    def _metric_cosine(self, X, X_samples):
        """
        Implements euclidean distance, computes the distance from each X_sample to
        all X individuals.
        X_samples row. Assuming numeric values.
        :param X: Matrix (N_train x M) (Individuals x Features)
        :param X_samples: Matrix (N_samples x M) (Individuals x Features)
        :return: Matrix of cosine distances (N_samples x N_train)
        """
        dist_mat = np.zeros((X_samples.shape[0], X.shape[0]))
        for j in np.arange(X_samples.shape[0]):
            if self.weights is None:
                Xw = X
                X_samples_w = X_samples[j, :]
            else:
                Xw = X * self.weights
                X_samples_w = X_samples[j, :] * self.weights

            product = np.dot(Xw, X_samples_w.transpose())
            modul1 = np.nansum(np.power(Xw, 2), axis=1)
            modul2 = np.nansum(np.power(X_samples_w, 2))
            suma1 = (modul1 + modul2)
            # div=product/suma1
            dist_vec = 1 - product / (modul1 + modul2)
            dist_mat[j, :] = dist_vec

        return dist_mat

    def _metric_minkowski(self, X, X_samples):
        """
        Implements euclidean distance, computes the distance from each X_sample to
        all X individuals.
        X_samples row. Assuming numeric values.
        :param X: Matrix (N_train x M) (Individuals x Features)
        :param X_samples: Matrix (N_samples x M) (Individuals x Features)
        :param p distance exponent
        :return: Matrix of minkowski distances (N_samples x N_train)
        """
        p = self.p
        dist_mat = np.zeros((X_samples.shape[0], X.shape[0]))
        for j in np.arange(X_samples.shape[0]):
            if self.weights is None:
                diff = X - X_samples[j, :]
            else:
                diff = X * self.weights - X_samples[j, :] * self.weights
            dist_vec = np.power(np.nansum(np.power(diff, p), axis=1), 1. / p)
            dist_mat[j, :] = dist_vec

        return dist_mat

    def _metric_correlation(self, X, X_samples):
        """
        Implements euclidean distance, computes the distance from each X_sample to
        all X individuals.
        X_samples row. Assuming numeric values.
        :param X: Matrix (N_train x M) (Individuals x Features)
        :param X_samples: Matrix (N_samples x M) (Individuals x Features)
        :return: Matrix of euclidean distances (N_samples x N_train)
        """

        dist_mat = np.zeros((X_samples.shape[0], X.shape[0]))
        for j in np.arange(X_samples.shape[0]):
            if self.weights is None:
                Xw = X
                X_samples_w = X_samples[j, :]
            else:
                Xw = X * self.weights
                X_samples_w = X_samples[j, :] * self.weights
            dist_mat[j, :] = [sp.spatial.distance.correlation(x, X_samples_w) for x in Xw]
        #     diff = X - X_samples[j,:]
        #     dist_vec = np.sqrt(np.nansum(np.power(diff, 2), axis=1))
        #     dist_mat[j,:] = dist_vec

        return dist_mat

    def fit(self, train_X, train_y):
        """
        Stores the training values for future predictions
        :param train_X: training matrix of features (N_train x M_features)
        :param train_y:  training vector of lables (N_train x 1)
        """
        self.train_X = train_X
        self.train_y = train_y

    def predict(self, X_samples):
        """
        Performs a prediction by running the k-Nearest-Neightbour algorithm using
        the selected metric and policy functions.
            1) Computes the distances between train set and the X_samples
            2) Compute the rank of the distance matrix row wise
            3) Compute the prediction using the rank matrix
        :param X_samples: matrix of features to be predicted (N_samples x M_features)
        :return: triple containing prediction (N_samples x 1),
            distance matrix (N_samples x K) and indexes (N_samples x K)
        """
        dist_mat = self.metric_fun(self.train_X, X_samples)
        dist_idx_mat = np.argsort(dist_mat)
        pred_vec = self.policy_fun(dist_idx_mat)
        return pred_vec
        # , np.sort(dist_mat, axis=1)[:, np.arange(self.k)], dist_idx_mat[:, np.arange(self.k)])
