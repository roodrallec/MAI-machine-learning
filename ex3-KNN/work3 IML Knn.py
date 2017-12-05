# coding=utf-8
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

            #if 'Class' not in label and 'class' not in label and 'a17' not in label:
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

    # input varaibles in rows, features in columns (x1 .... xn)
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
"""3 Write a Python function for classifying, using a KNN algorithm, each instance from the TestMatrix using the
TrainMatrix to a classifier called kNNAlgorithm(…). You decide the parameters for this classifier.
Justify your implementation and add all the references you have considered for your decisions. K IS A HYPER PARAMETER"""

def kNNAlgorithm(x_train, x_train_class, x_test, k_neiggbours=1, method='', dist_meas='euclidean'):


    j = 0

    test_results = np.empty(x_test.shape[0])

    for test_x in x_test:

        i=0
        distances = np.empty(x_train_class.shape)

        for train_x in x_train:
            #print(test_x.shape)
            #print(train_x.shape)

            """
            4
            For the similarity function, you should consider the Hamming, Euclidean, Cosine,
            and another EXTRA (you decide which one) distance functions. Adapt these distances to handle
            all kind of attributes (i.e., numerical and categorical).

            """

            ## Calculate distance deoending on selected measurement

            if dist_meas in 'euclidean':
                distances[ i ] = LA.norm(test_x - train_x)
            if dist_meas in 'hamming':
                distances[ i ]=sp.spatial.distance.hamming(test_x, train_x)
            if dist_meas in 'cosine':
                distances[ i ] = sp.spatial.distance.cosine(test_x, train_x)
            if dist_meas in 'mahalanobis':
                distances[ i ] = sp.spatial.distance.mahalanobis(test_x, train_x, LA.inv(np.cov([test_x,train_x], rowvar=False)))
            if dist_meas in 'minkowski':
                distances[ i ] = sp.spatial.distance.minkowski(test_x, train_x, p=4)
            if dist_meas in 'correlation':
                    distances[ i ] = sp.spatial.distance.correlation(test_x, train_x)

            i+= 1

        ## if there is an exact match, we assign the class of such match

        if 0: #in distances:

            selected_neighbours = np.where(distances == 0)[ 0 ]
            selected_neighbours_class=x_train_class[selected_neighbours]
            test_results[j] = selected_neighbours_class


        else:

        ## we assigned the class based on the K neighbours


            idx = distances.argsort()
            ordered_class = x_train_class[ idx ]
            selected_neighbours_class = ordered_class[ 0:k_neiggbours ]

            ordered_neighbours = x_train[ idx ]
            selected_neighbours = ordered_neighbours[ 0:k_neiggbours  ]


            """ a. To decide the solution of the current_instance, you may consider using two policies:
            the most similar retrieved case and a voting policy."""
            test_results[ j ] = selected_neighbours_class[ 0 ]  # most similar

            if method in 'vote':
                c = Counter([t for t in selected_neighbours_class])
                voting_result= c.most_common(1)[ 0 ][ 0 ]
                test_results[j] = voting_result

        j+= 1

    """Assume that the KNN algorithm returns the K most similar instances (i.e., also known as cases)
    from the TrainMatrix to the current_instance. The value of K will be setup in your evaluation to 1, 3, 5, and 7."""
    return test_results, selected_neighbours, selected_neighbours_class, ordered_neighbours, ordered_class


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
                'datasetsCBR/' + dataset_name + '/' + dataset_name + '.fold.00000' + str(f) + '.train.arff', classField, emptyNominalField )

            #TrainMatrix, mean_train, std_train = normalizeMeanSTD(TrainMatrix)
            #TrainMatrix, min_train, max_train = normalizeMinMax(TrainMatrix)

            # loading test data

            TestMatrix, test_x_labels, test_x_class, x_class_names = read_dataset(
                'datasetsCBR/' + dataset_name + '/' + dataset_name + '.fold.00000' + str(f) + '.test.arff', classField, emptyNominalField)

            #TestMatrix, mean_train, std_train = normalizeMeanSTD(TestMatrix)
            #TestMatrix, min_test, max_test = normalizeMinMax(TestMatrix)

            # RUN ALGORITHM
            t1 = datetime.now()
            prediction, k_neighbours, k_neighbours_class,_,_ = kNNAlgorithm(TrainMatrix, train_x_class, TestMatrix, k_neig, method=sel_method, dist_meas=dist)
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

#NEED TO record processing time