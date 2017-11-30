import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
np.random.seed(0)


# Using Iris for testing the development
iris = datasets.load_iris()
iris_sample_idx = np.random.choice(np.arange(iris.data.shape[0]), 15)

train_X = iris.data[iris_sample_idx[:10], :2]
train_y = iris.target[iris_sample_idx[:10]]
test_X = iris.data[iris_sample_idx[10:], :2]
test_y = iris.target[iris_sample_idx[10:]]

# Check Scikit-learn output
clf = neighbors.KNeighborsClassifier(3)
clf.fit(train_X, train_y)
dist, idx = clf.kneighbors(train_X)
pred_y = clf.predict(test_X)
pred_y == test_y

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

hepa_mat_names = np.array(hepa.dtype.names)
hepa_mat = np.array([hepa[n] for n in hepa_mat_names]).transpose()
hepa_mat = as_numeric(hepa_mat)
# Do any other data imputation or manipulation here

# For Developing
hepa_train_X = hepa_mat[:10, :19]
hepa_train_y = hepa_mat[:10, 20]
hepa_test_X = hepa_mat[11:13, :19]
hepa_test_y = hepa_mat[11:13, 20]

# WIP
dist = np.sqrt(np.nansum(np.power(hepa_mat[0] - hepa_mat,2),axis=1))


class kNNAlgorithm(object):
    def __init__(self, k, metric="euclidean", policy="voting"):
        self.k = k
        self.metric = metric
        self.policy = policy

    # Hamming, Euclidean, Cosine and
    # def _diff(self, x, y):
    #     """
    #     Calculate the difference between two values. If both are numeric, the
    #     minus operator is used (diff = x - y). If both are categorical the
    #     equal operator is used (diff = 0 if x == y, diff = 1 otherwise)
    #     used. The
    #     :param x: value (numeric or string)
    #     :param y: value (numeric or string)
    #     :return: difference
    #     """
    #     if hasattr(x, '__sub__') and hasattr(y, '__sub__'):  # Numeric
    #         return x - y
    #     else:  # Categorical
    #         if x == y:
    #             return 0.0
    #         else:
    #             return 1.0

    # def _metric_euclidean(self, x, y):
    #     diffs = np.array([self._diff(e1, e2) for e1, e2 in zip(x, y)])
    #     return np.sqrt(np.sum(np.power(diffs, 2)))

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y


knn = kNNAlgorithm(3)
knn._metric_euclidean(np.array([3,5.0,-2]),np.array([1,0,-1]))