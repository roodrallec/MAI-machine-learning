import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
from numpy import linalg as LA

np.set_printoptions(linewidth=120)

"""1. Improve the parser developed in previous works in order to use the class attribute, too.

"""


## READING FILES
def read_dataset(fileroute):
    global x_class, x_class_names
    x, x_meta = arff.loadarff(fileroute)

    x_labels = x_meta.names()

    x_allnumeric = np.empty([x.size, x_labels.__len__() - 1])

    i = 0

    for label in x_labels:

        if 'nominal' in x_meta[label][0]:

            nominal_values, numeric_eq = np.unique(x[label], return_inverse=True)

            if 'Class' not in label and 'class' not in label and 'a17' not in label:
                x_allnumeric[:, i] = numeric_eq
                i += 1
            else:
                x_class = numeric_eq
                x_class_names = nominal_values

        else:
            x_allnumeric[:, i] = x[label]
            i += 1

    # input varaibles in rows, features in columns (x1 .... xn)
    return x_allnumeric, x_labels, x_class, x_class_names


### Normalization
def normalizeMinMax(x):
    minVal = np.min(x, axis=0)
    maxVal = np.max(x, axis=0)
    norm_x = (x - minVal) / (maxVal - minVal)
    return norm_x, minVal, maxVal


def unnormalizeMinMax(norm_x, minVal, maxVal):
    return minVal + np.dot(norm_x, (maxVal - minVal))


def normalizeMeanSTD(x):
    means_array = np.mean(x, axis=0)
    std_array = np.std(x, axis=0)
    norm_x = (x - means_array) / std_array
    return norm_x, means_array, std_array


def unnormalizeMeanSTD(norm_x, std_array, means_array):
    return means_array + np.dot(norm_x, std_array)

#####################################################################################################################

def knn(x_train,x_train_labels,x_test,k_neiggbours='1',k_weights='1',method='vote', dist_meas='euclidian'):



    return x_test_labels

#####################################################################################################################

# SETTING THE DATASET

dataset_name = 'hepatitis'

# S-FOLD LOOP

"""
1 [cont] Now, you need to read and save the information from a training and their corresponding testing file in a
TrainMatrix and a TestMatrix, respectively. Recall that you need to normalize all the numerical attributes
 in the range [0..1].
2 Write a Python function that automatically repeats the process described in previous step for the
10-fold cross-validation files. That is, read automatically each training case and run each one of the test cases in
the selected classifier.
"""
for i in range(0, 10):
    # loading training data

    TrainMatrix, x_labels, x_class, x_class_names = read_dataset(
        'datasetsCBR/' + dataset_name + '/' + dataset_name + '.fold.00000' + str(i) + '.train.arff')

    TrainMatrix, mean, std = normalizeMeanSTD(TrainMatrix)
    TrainMatrix, min, max = normalizeMinMax(TrainMatrix)


    # loading test data

    TestMatrix, x_labels, x_class, x_class_names = read_dataset(
        'datasetsCBR/' + dataset_name + '/' + dataset_name + '.fold.00000' + str(i) + '.test.arff')

    # RUN ALGORITHM


    # CALCULATE & SAVE ACCURACY FOR THIS FOLD

print('done')

