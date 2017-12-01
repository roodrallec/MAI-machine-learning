import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt
from numpy import linalg as LA

np.set_printoptions(linewidth=120)


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

            if 'Class' not in label and 'class' not in label:
                x_allnumeric[:, i] = numeric_eq
                i += 1
            else:
                x_class = numeric_eq
                x_class_names = nominal_values

        else:
            x_allnumeric[:, i] = x[label]
            i += 1

    #input varaibles in rows, features in columns (x1 .... xn)
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

#SETTING THE DATASET

dataset_name='credit-a'


#S-FOLD LOOP

for i in range(0,10):
    #loading training data

    x, x_labels, x_class, x_class_names = read_dataset('datasetsCBR/' + dataset_name + '/' + dataset_name + '.fold.00000'+str(i)+'.train.arff')

    #RUN ALGORITHM

    #CALCULATE & SAVE ACCURACY FOR THIS FOLD

    #loading test data

    x, x_labels, x_class, x_class_names = read_dataset(
        'datasetsCBR/' + dataset_name + '/' + dataset_name + '.fold.00000' + str(i) + '.test.arff')


print('done')