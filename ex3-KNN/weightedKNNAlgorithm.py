import sklearn.feature_selection as skfs
from kNNAlgorithm import *

def weightedKNNAlgorithm(X_train,y_train,k, dist,weight_method=None) :

    weights=None

    if weight_method is "info_gain":
        weights=skfs.mutual_info_classif(X_train,y_train)

    #print (weights)

    algo = kNNAlgorithm(k, metric=dist, p=4, policy='voting', weights=weights, selection=None)


    return algo