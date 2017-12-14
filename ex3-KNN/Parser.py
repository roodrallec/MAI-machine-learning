# coding=utf-8
"""
    Improve the parser developed in previous works in order to use the class attribute, too.
"""
import numpy as np
from scipy.io import arff
from collections import Counter


def read_dataset(fileroute, classfield='class', emptyNomField='?'):
    """ Loads dataset information from file, computes values for missing features, creates an all numeric matrix with
    all the instances features.
    Creates a array containing the features names
    Creates a array containing the class names
    Creates a array containing the class values.

    parameters:
    fileroute: path to dataset file
    classfield: name of the field containing the class value
    emptyNomField: characters used in empty (missing information) in nominal features

    returns:
    x_allnumeric: n (number of instances) x m (number of features) numpy matrix
    x_labels: 1x m array containing features labels
    x_class: 1 x n array containing class assignment for each instance
    x_class_names: 1 x c (number of different classes) array containing the names of the classes
    """
    global x_class, x_class_names
    x, x_meta = arff.loadarff(fileroute)

    x_labels = x_meta.names()

    x_allnumeric = np.empty([x.size, x_labels.__len__() - 1])

    i = 0

    for label in x_labels:

        if 'nominal' in x_meta[label][0]:

            if emptyNomField not in x[label]:
                c = Counter([t for t in x[label]])
                most_c = c.most_common(1)[0][0]

                if most_c in '?':
                    most_c = c.most_common(2)[1]
                idx = np.where(x[label] == '?')[0]
                x[label][idx] = most_c

            nominal_values, numeric_eq = np.unique(x[label], return_inverse=True)

            if classfield not in label:
                x_allnumeric[:, i] = numeric_eq
                i += 1
            else:
                x_class = numeric_eq
                x_class_names = nominal_values

        else:
            x_allnumeric[:, i] = x[label]
            nan_idx = np.where(np.isnan(x_allnumeric[:, i]))[0]
            if nan_idx.size > 0:
                x_allnumeric[nan_idx, i] = np.nanmean(x_allnumeric[:, i])
            i += 1

    # input variables in rows, features in columns (x1 .... xn)
    return x_allnumeric, x_labels, x_class, x_class_names


# Normalization
def normalize_min_max(x):
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)
    idx_zeros = np.where((min_val - max_val) == 0)[0]
    min_val[idx_zeros] = 0
    idx_zeros_max_v = [idxz for idxz in idx_zeros if max_val[idxz] == 0]
    max_val[idx_zeros_max_v] = 1
    norm_x = (x - min_val) / (max_val - min_val)
    return norm_x, min_val, max_val


def un_normalize_min_max(norm_x, minVal, maxVal):
    return minVal + np.dot(norm_x, (maxVal - minVal))


def normalize_mean_std(x):
    means_array = np.mean(x, axis=0)
    std_array = np.std(x, axis=0)
    idx_zeros = np.where(std_array == 0)[0]
    std_array[idx_zeros] = 1
    norm_x = (x - means_array) / std_array
    return norm_x, means_array, std_array


def un_normalize_mean_std(norm_x, std_array, means_array):
    return means_array + np.dot(norm_x, std_array)
