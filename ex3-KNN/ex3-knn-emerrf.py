# coding=utf-8
from sklearn import neighbors, datasets
import numpy as np
from scipy.io import arff
from collections import Counter
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy as sp
import sklearn
from sklearn import metrics as skmetrics
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime


np.set_printoptions(linewidth=120)
np.random.seed(0)

# Note: Just do the checks for K = {1, 3, 5 and 7}

# Load data (numerical + categorical)
# As we have binary categories, we can works with 1 and 0, directly
# hepa, hepa_meta = arff.loadarff(
#     "datasets/hepatitis/hepatitis.fold.000000.train.arff")
#
#
# def as_numeric(value):
#     """
#     Function that parses Hepatitis dataset values to numeric. First tries to
#     return a float. If string is detected, then returns categorical encoding
#     (0 or 1).
#     :param value: string
#     :return: float
#     """
#     try:
#         return np.float(value)
#     except ValueError:
#         if value in ['male', 'no', 'DIE']:
#             return np.float(0.0)
#         elif value in ['female', 'yes', 'LIVE']:
#             return np.float(1.0)
#         else:
#             return np.nan
#
# as_numeric = np.vectorize(as_numeric, otypes=[np.float])
# # Ignore warning from vectorize as we are generating nan on purpose...
# # RuntimeWarning: invalid value encountered in as_numeric (vectorized)
# hepa_mat_names = np.array(hepa.dtype.names)
# hepa_mat = np.array([hepa[n] for n in hepa_mat_names]).transpose()
# hepa_mat = as_numeric(hepa_mat)
# # Quick and dirty nan imputation
# for j in np.arange(hepa_mat.shape[1]):
#     nan_idx = np.where(np.isnan(hepa_mat[:, j]))[0]
#     if nan_idx.size > 0:
#         hepa_mat[nan_idx, j] = np.nanmean(hepa_mat[:, j])
#
# # For Developing
# hepa_train_X = hepa_mat[:10, :19]
# hepa_train_y = hepa_mat[:10, 19]
# hepa_test_X = hepa_mat[11:13, :19]
# hepa_test_y = hepa_mat[11:13, 19]



#####################################################################################################################
"""3 Write a Python function for classifying, using a KNN algorithm, each instance from the TestMatrix using the
TrainMatrix to a classifier called kNNAlgorithm(…). You decide the parameters for this classifier.
Justify your implementation and add all the references you have considered for your decisions. K IS A HYPER PARAMETER"""


class kNNAlgorithm(object):
    def __init__(self, k, metric="euclidean", q=4, policy="voting"):
        self.k = k
        self._select_metric_function(metric,q)
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

    def _select_metric_function(self, metric,q):
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
            diff = X - X_samples[j,:]
            dist_vec = np.sqrt(np.nansum(np.power(diff, 2), axis=1))
            dist_mat[j,:] = dist_vec

        return dist_mat

    def _metric_minkowski(self, X, X_samples):
        """
        Implements euclidean distance, computes the distance from each X_sample to
        all X individuals.
        X_samples row. Assuming numeric values.
        :param X: Matrix (N_train x M) (Individuals x Features)
        :param X_samples: Matrix (N_samples x M) (Individuals x Features)
        :param q distance exponent
        :return: Matrix of minkowski distances (N_samples x N_train)
        """
        dist_mat = np.zeros((X_samples.shape[0], X.shape[0]))
        for j in np.arange(X_samples.shape[0]):
            diff = X - X_samples[j,:]
            dist_vec = np.sqrt(np.nansum(np.power(diff, 2), axis=1))
            dist_mat[j,:] = dist_vec

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
            diff = X - X_samples[j,:]
            dist_vec = np.sqrt(np.nansum(np.power(diff, 2), axis=1))
            dist_mat[j,:] = dist_vec

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


# # Run Algorithm for a tiny hepa
# knn = kNNAlgorithm(3, metric="hamming")
# knn.fit(hepa_train_X, hepa_train_y)
# our_pred, our_dist, our_idx = knn.predict(hepa_test_X)
#
# # Check Scikit-learn output
# clf = neighbors.KNeighborsClassifier(3)
# clf.fit(hepa_train_X, hepa_train_y)
# dist, idx = clf.kneighbors(hepa_test_X)
# pred_y = clf.predict(hepa_test_X)
#
# print our_pred == pred_y
# print our_dist == dist
# print our_idx == idx




"""1. Improve the parser developed in previous works in order to use the class attribute, too.

"""


## READING FILES
def read_dataset(fileroute, classfield='class', emptyNomField='?' ):
    global x_class, x_class_names
    x, x_meta = arff.loadarff(fileroute)

    x_labels = x_meta.names()

    x_allnumeric = np.empty([ x.size, x_labels.__len__() - 1 ])

    i = 0

    for label in x_labels:

        if 'nominal' in x_meta[ label ][ 0 ]:

            if emptyNomField not in x[ label ]:
                c = Counter([ t for t in x[ label ] ])
                most_c = c.most_common(1)[ 0 ][ 0 ]
                if most_c in '?':
                    most_c = c.most_common(2)[ 1 ]
                idx = np.where(x[ label ] == '?')[ 0 ]
                x[ label ][ idx ] = most_c

            nominal_values, numeric_eq = np.unique(x[ label ], return_inverse=True)


            if classfield not in label:
                x_allnumeric[ :, i ] = numeric_eq
                i += 1
            else:
                x_class = numeric_eq
                x_class_names = nominal_values

        else:
            x_allnumeric[ :, i ] = x[ label ]
            nan_idx = np.where(np.isnan(x_allnumeric[ :, i ]))[ 0 ]
            if nan_idx.size > 0:
                x_allnumeric[ nan_idx, i ] = np.nanmean(x_allnumeric[ :, i ])
            i += 1

    # input variables in rows, features in columns (x1 .... xn)
    return x_allnumeric, x_labels, x_class, x_class_names


### Normalization
def normalizeMinMax(x):
    minVal = np.min(x, axis=0)
    maxVal = np.max(x, axis=0)

    idx_zeros = np.where((minVal - maxVal) == 0)[0]
    minVal[idx_zeros]=0



    idx_zeros_maxV=[idxz for idxz in idx_zeros if maxVal[idxz] == 0]
    maxVal[ idx_zeros_maxV ] = 1

    norm_x = (x - minVal) / (maxVal - minVal)
    return norm_x, minVal, maxVal


def unnormalizeMinMax(norm_x, minVal, maxVal):
    return minVal + np.dot(norm_x, (maxVal - minVal))


def normalizeMeanSTD(x):
    means_array = np.mean(x, axis=0)
    std_array = np.std(x, axis=0)
    idx_zeros = np.where(std_array == 0)[ 0 ]
    std_array[ idx_zeros ] = 1
    norm_x = (x - means_array) / std_array
    return norm_x, means_array, std_array


def unnormalizeMeanSTD(norm_x, std_array, means_array):
    return means_array + np.dot(norm_x, std_array)



#####################################################################################################################

# SETTING THE DATASET

dataset_name = 'hepatitis' #pen-based
emptyNominalField='?'
classField='Class'

dataset_name = 'pen-based' #pen-based
emptyNominalField=''
classField='a17'

# S-FOLD LOOP

"""
1 [cont] Now, you need to read and save the information from a training and their corresponding testing file in a
TrainMatrix and a TestMatrix, respectively. Recall that you need to normalize all the numerical attributes
 in the range [0..1].
2 Write a Python function that automatically repeats the process described in previous step for the
10-fold cross-validation files. That is, read automatically each training case and run each one of the test cases in
the selected classifier.
"""
sel_method='vote'
results_distance=[]

print('STARTTING')

for dist in ['euclidean','cosine', 'hamming', 'minkowski', 'correlation']: #in ['euclidean', 'cosine', 'hamming', 'mahalanobis', 'minkowski']:

    results_k=[]
    print(dist)

    for k_neig in [ 1, 3, 5, 7 ]: #used to be k... takes to long to calculate same every time so we call 7 and then iterate for each

        fold_results = np.empty([10, 4])

        for f in range(0, 10):

            # loading training data

            TrainMatrix, train_x_labels, train_x_class, x_class_names = read_dataset(
                'datasets/' + dataset_name + '/' + dataset_name + '.fold.00000' + str(f) + '.train.arff', classField, emptyNominalField )

            TrainMatrix, mean_train, std_train = normalizeMeanSTD(TrainMatrix)
            #TrainMatrix, min_train, max_train = normalizeMinMax(TrainMatrix)

            # loading test data

            TestMatrix, test_x_labels, test_x_class, x_class_names = read_dataset(
                'datasets/' + dataset_name + '/' + dataset_name + '.fold.00000' + str(f) + '.test.arff', classField, emptyNominalField)

            TestMatrix, mean_train, std_train = normalizeMeanSTD(TestMatrix)
            #TestMatrix, min_test, max_test = normalizeMinMax(TestMatrix)

            # RUN ALGORITHM
            t1 = datetime.now()
            #prediction, k_neighbours, k_neighbours_class,_,_ = kNNAlgorithm(TrainMatrix, train_x_class, TestMatrix, k_neig, method=sel_method, dist_meas=dist)
            # Run Algorithm for a tiny hepa
            knn = kNNAlgorithm(k_neig, metric=dist)
            knn.fit(TrainMatrix, train_x_class)
            prediction, our_dist, our_idx = knn.predict(TestMatrix)
            t2 = datetime.now()
            delta = t2 - t1



            """b. For evaluating the performance of the KNN algorithm, we will use the percentage of
            correctly classified instances. To this end, at least, you should store the number of cases correctly
            classified, the number of cases incorrectly classified. This information will be used for the
            evaluation of the algorithm. You can store your results in a memory data structure or in a file.
            Keep in mind that you need to compute the average accuracy over the 10-fold cross-validation sets."""



            # CALCULATE & SAVE ACCURACY FOR THIS FOLD
            knn_accuracy=skmetrics.accuracy_score(test_x_class, prediction)
            knn_correct_classified_samples = skmetrics.accuracy_score(test_x_class, prediction, normalize=False)
            knn_incorrect_classified_samples = TestMatrix.shape[0]-knn_correct_classified_samples

            t3 = datetime.now()
            neigh = KNeighborsClassifier(n_neighbors=k_neig, metric=dist, p=4)


            neigh.fit(TrainMatrix, train_x_class)

            prediction_sk=neigh.predict(TestMatrix)
            t4 = datetime.now()
            delta2 = t4 - t3
            sk_knn_accuracy = skmetrics.accuracy_score(test_x_class, prediction_sk)

            fold_results[ f ] = [ knn_accuracy, knn_correct_classified_samples, knn_incorrect_classified_samples, sk_knn_accuracy ]

            if f==0:
                print ("Fold accuracy result comparison for k={0}, distance = {1} and method = {2}".format(k_neig, dist, sel_method))
                print('Prediction KNN IML   -   Prediction SKLearn   -- timeKNNIML -- timeSKlearn -- VECTOR KNN IML  // VECTOR KNN SKLEARN  //  TRUTH')
            print('{0:.3f}          {1:.3f}       {5:.3f}   {6:.3f}     {2}  //  {3}   //  {4}'.format(knn_accuracy,sk_knn_accuracy, prediction, prediction_sk, test_x_class, delta.total_seconds(),delta2.total_seconds()))

        knn_avg_accuracy=np.mean(fold_results[:, 0])
        print('Average accuracy kNNAlgorithm {0:.3f}  SKlearn Knn {1:.3f} '.format(knn_avg_accuracy,np.mean(fold_results[:, 3]) ))

        results_k.append([k_neig, knn_avg_accuracy, fold_results])

    print('Avarage accuracies for k 1,3,5,7: distance = {0} and method = {1}'.format(dist, sel_method))
    print('k    1       3       5       7')
    print('   {0:.3f}  {1:.3f}   {2:.3f}   {3:.3f}'.format(results_k[0][1],results_k[1][1],results_k[2][1],results_k[3][1]))
    results_distance.append([dist, results_k])

print('Avarage accuracies for k 1,3,5,7: and method = {0}'.format(sel_method))
print('k      1         3         5         7')
print('{4}   {0:.3f}  {1:.3f}   {2:.3f}   {3:.3f}'.format(results_distance[0][1][0][1],results_distance[0][1][1][1],results_distance[0][1][2][1],results_distance[0][1][3][1], results_distance[0][0]))
print('{4}   {0:.3f}  {1:.3f}   {2:.3f}   {3:.3f}'.format(results_distance[1][1][0][1],results_distance[1][1][1][1],results_distance[1][1][2][1],results_distance[1][1][3][1], results_distance[1][0]))
print('{4}   {0:.3f}  {1:.3f}   {2:.3f}   {3:.3f}'.format(results_distance[2][1][0][1],results_distance[2][1][1][1],results_distance[2][1][2][1],results_distance[2][1][3][1], results_distance[2][0]))
print('{4}   {0:.3f}  {1:.3f}   {2:.3f}   {3:.3f}'.format(results_distance[3][1][0][1],results_distance[3][1][1][1],results_distance[3][1][2][1],results_distance[3][1][3][1], results_distance[3][0]))
print('{4}   {0:.3f}  {1:.3f}   {2:.3f}   {3:.3f}'.format(results_distance[4][1][0][1],results_distance[4][1][1][1],results_distance[4][1][2][1],results_distance[4][1][3][1], results_distance[4][0]))

print('\n\ndone')

