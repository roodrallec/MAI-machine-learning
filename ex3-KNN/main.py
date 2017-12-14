"""
    1 [cont] Now, you need to read and save the information from a training and their corresponding testing file in a
    TrainMatrix and a TestMatrix, respectively. Recall that you need to normalize all the numerical attributes
     in the range [0..1].
    2 Write a Python function that automatically repeats the process described in previous step for the
    10-fold cross-validation files. That is, read automatically each training case and run each one of the test cases in
    the selected classifier.

    b. For evaluating the performance of the KNN algorithm, we will use the percentage of
    correctly classified instances. To this end, at least, you should store the number of cases correctly
    classified, the number of cases incorrectly classified. This information will be used for the
    evaluation of the algorithm. You can store your results in a memory data structure or in a file.
    Keep in mind that you need to compute the average accuracy over the 10-fold cross-validation sets.
"""
import sys
import numpy as np
from kNNAlgorithm import *
from Parser import *
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
np.set_printoptions(linewidth=120)


def load_and_normalise(path, class_field, dummy_nominal):
    matrix, labels, x_class, _ = read_dataset(path, class_field, dummy_nominal)
    matrix, min_train, max_train = normalize_min_max(matrix)
    return matrix, x_class


def run_knn(knn, X_train, y_train, X_test):
    t1 = datetime.now()
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    t2 = datetime.now()
    delta = (t2 - t1).total_seconds()
    return prediction, delta


def run_fold_knn(fold_path, class_field, empty_nominal_field):
    # Load and normalise training data and test data
    X_train, y_train = load_and_normalise(fold_path + '.train.arff', class_field, empty_nominal_field)
    X_test, x_labels = load_and_normalise(fold_path + '.test.arff', class_field, empty_nominal_field)
    # Run the two implementations
    y, delta = run_knn(kNNAlgorithm(k, metric=dist, p=4), X_train, y_train, X_test)
    y_sk, delta_sk = run_knn(KNeighborsClassifier(n_neighbors=k, metric=dist), X_train, y_train, X_test)
    # Calculate the two accuracies
    knn_accuracy = accuracy_score(x_labels, y)
    sk_knn_accuracy = accuracy_score(x_labels, y_sk)
    # correct and incorrect samplesZ
    knn_correct = accuracy_score(x_labels, y, normalize=False)
    knn_incorrect = X_test.shape[0] - knn_correct
    print('{0:.3f}\t{1:.3f}\t{5:.3f}\t{6:.3f}\t{2} // {3} // {4}'
          .format(knn_accuracy, sk_knn_accuracy, y, y_sk, x_labels, delta, delta_sk))
    return [knn_accuracy, knn_correct, knn_incorrect, sk_knn_accuracy]

# Hep Data-set
dataset_name = 'hepatitis'  # pen-based
empty_nominal_field = '?'
class_field = 'Class'

# Pen-based Data-set
# dataset_name = 'pen-based' #pen-based
# emptyNominalField=''
# classField='a17'

# S-FOLD LOOP
sel_method = 'vote'
res_dist = []

orig_stdout = sys.stdout
out_file = open('out_weighted_no_weight.txt', 'w')
# sys.stdout = out_file

for dist in ['euclidean', 'cosine', 'hamming', 'minkowski', 'correlation']: # 'mahalanobis'
    results_k = []
    print(dist)

    for k in [1, 3, 5, 7]: # k takes to long to calculate and the same every time so we call 7 and then iterate.
        fold_results = np.empty([10, 4])

        print ("Fold accuracy result comparison for k={0}, distance = {1} and method = {2}".format(k, dist,
                                                                                                   sel_method))
        print('Prediction KNN IML   -   Prediction SKLearn   -- timeKNNIML -- timeSKlearn -- VECTOR KNN IML // '
              'VECTOR KNN SK-LEARN // TRUTH')

        for f in range(0, 10):
            fold_path = 'datasets/{0}/{0}.fold.00000{1}'.format(dataset_name, f)
            fold_results[f] = run_fold_knn(fold_path, class_field, empty_nominal_field)

        knn_avg_accuracy = np.mean(fold_results[:, 0])
        sklearn_mean = np.mean(fold_results[:, 3])
        print('Average accuracy kNNAlgorithm {0:.3f} SKlearn Knn {1:.3f} '.format(knn_avg_accuracy, sklearn_mean))
        results_k.append([k, knn_avg_accuracy, fold_results])

    print('Avarage accuracies for k 1,3,5,7: distance = {0} and method = {1}'.format(dist, sel_method))
    print('k\t1\t3\t5\t7')
    print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(results_k[0][1], results_k[1][1], results_k[2][1],
                                                      results_k[3][1]))
    res_dist.append([dist, results_k])

print('Average accuracies for k 1,3,5,7: and method = {0}'.format(sel_method))
print('k\t1\t3\t5\t7')
print('{4}\t{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(res_dist[0][1][0][1], res_dist[0][1][1][1], res_dist[0][1][2][1],
                                                       res_dist[0][1][3][1], res_dist[0][0]))
print('{4}\t{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(res_dist[1][1][0][1], res_dist[1][1][1][1], res_dist[1][1][2][1],
                                                       res_dist[1][1][3][1], res_dist[1][0]))
print('{4}\t{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(res_dist[2][1][0][1], res_dist[2][1][1][1], res_dist[2][1][2][1],
                                                       res_dist[2][1][3][1], res_dist[2][0]))
print('{4}\t{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(res_dist[3][1][0][1], res_dist[3][1][1][1], res_dist[3][1][2][1],
                                                       res_dist[3][1][3][1], res_dist[3][0]))
print('{4}\t{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'.format(res_dist[4][1][0][1], res_dist[4][1][1][1], res_dist[4][1][2][1],
                                                       res_dist[4][1][3][1], res_dist[4][0]))
print('\n\ndone')

sys.stdout = orig_stdout
out_file.close()