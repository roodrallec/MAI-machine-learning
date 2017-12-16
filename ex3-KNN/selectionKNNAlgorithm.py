import sklearn.feature_selection as skfs
from kNNAlgorithm import *


def selectionKNNAlgorithm(X_train, y_train, k, dist, selection_method=None, number_features=None):

    weights = None

    if selection_method is "info_gain":
        weights = skfs.mutual_info_classif(X_train, y_train)

    print (weights)
    if number_features is not None:
        w_idx=np.argsort(weights)
        weights[ weights < weights[ w_idx[-number_features] ] ] = 0
        weights[ weights >= weights[ w_idx[-number_features] ] ]= 1

    #print (weights)

    algo = kNNAlgorithm(k, metric=dist, p=4, policy='voting', weights=weights, selection=None)

    return algo