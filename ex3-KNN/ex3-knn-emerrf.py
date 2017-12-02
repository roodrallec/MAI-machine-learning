import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
np.random.seed(0)

# Note: Just do the checks for K = {1, 3, 5 and 7}

# Load data (numerical + categorical)
# As we have binary categories, we can works with 1 and 0, directly
hepa, hepa_meta = arff.loadarff(
    "datasets/hepatitis/hepatitis.fold.000000.train.arff")


def as_numeric(value):
    """
    Function that parses Hepatitis dataset values to numeric. First tries to
    return a float. If string is detected, then returns categorical encoding
    (0 or 1).
    :param value: string
    :return: float
    """
    try:
        return np.float(value)
    except ValueError:
        if value in ['male', 'no', 'DIE']:
            return np.float(0.0)
        elif value in ['female', 'yes', 'LIVE']:
            return np.float(1.0)
        else:
            return np.nan

as_numeric = np.vectorize(as_numeric, otypes=[np.float])
# Ignore warning from vectorize as we are generating nan on purpose...
# RuntimeWarning: invalid value encountered in as_numeric (vectorized)
hepa_mat_names = np.array(hepa.dtype.names)
hepa_mat = np.array([hepa[n] for n in hepa_mat_names]).transpose()
hepa_mat = as_numeric(hepa_mat)
# Quick and dirty nan imputation
for j in np.arange(hepa_mat.shape[1]):
    nan_idx = np.where(np.isnan(hepa_mat[:, j]))[0]
    if nan_idx.size > 0:
        hepa_mat[nan_idx, j] = np.nanmean(hepa_mat[:, j])

# For Developing
hepa_train_X = hepa_mat[:10, :19]
hepa_train_y = hepa_mat[:10, 19]
hepa_test_X = hepa_mat[11:13, :19]
hepa_test_y = hepa_mat[11:13, 19]


class kNNAlgorithm(object):
    def __init__(self, k, metric="euclidean", policy="nearest"):
        self.k = k
        self._select_metric_function(metric)
        self._select_policy_function(policy)

    def _select_policy_function(self, policy):
        """
        Selector function for setting the policy function.
        :param policy: name of the policy, one of "voting", "nearest"
        """
        if policy == "voting":
            self.policy_fun = self._policy_voting
        else:
            self.policy_fun = self._policy_nearest

    def _select_metric_function(self, metric):
        """
        Selector function for setting the metric function
        :param metric: name of the metric, one of "hamming", "cosine", "other",
        "euclidean"
        """
        if metric == "hamming":
            self.metric_fun = self._metric_hamming
        elif metric == "cosine":
            pass
        elif metric == "other":
            pass
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
        freq[start] = np.hstack([np.diff(start), xs.size-start[-1]])
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
            diff = X - X_samples[j,:]
            dist_vec = np.sqrt(np.nansum(np.power(diff, 2), axis=1))
            dist_mat[j,:] = dist_vec

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
            dist_mat[j, :] = np.mean(X != X_samples[j, :], axis=1)

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
        return (pred_vec,
                np.sort(dist_mat, axis=1)[:, np.arange(self.k)],
                dist_idx_mat[:, np.arange(self.k)])


# Run Algorithm for a tiny hepa
knn = kNNAlgorithm(3, metric="hamming")
knn.fit(hepa_train_X, hepa_train_y)
our_pred, our_dist, our_idx = knn.predict(hepa_test_X)

# Check Scikit-learn output
clf = neighbors.KNeighborsClassifier(3)
clf.fit(hepa_train_X, hepa_train_y)
dist, idx = clf.kneighbors(hepa_test_X)
pred_y = clf.predict(hepa_test_X)

print our_pred == pred_y
print our_dist == dist
print our_idx == idx