"""
    1. [cont] Now, you need to read and save the information from a training and their corresponding testing file in a
    TrainMatrix and a TestMatrix, respectively. Recall that you need to normalize all the numerical attributes
     in the range [0..1].

    2. Write a Python function that automatically repeats the process described in previous step for the
    10-fold cross-validation files. That is, read automatically each training case and run each one of the test cases in
    the selected classifier.

    b. For evaluating the performance of the KNN algorithm, we will use the percentage of correctly classified
    instances. To this end, at least, you should store:
        - the number of cases correctly classified,
        - the number of cases incorrectly classified
    This information will be used for the evaluation of the algorithm.
    You can store your results in a memory data structure or in a file.
    Keep in mind that you need to compute the average accuracy over the 10-fold cross-validation sets.
"""
import pandas as pd
from kNNAlgorithm import *
from Parser import *
from sklearn.metrics import accuracy_score
from datetime import datetime


def load_and_normalise(path, class_field, dummy_nominal):
    matrix, labels, x_class, _ = read_dataset(path, class_field, dummy_nominal)
    matrix, min_train, max_train = normalize_min_max(matrix)
    return matrix, x_class


def run_knn(knnAlgorithm, X_train, y_train, X_test):
    t1 = datetime.now()
    knnAlgorithm.fit(X_train, y_train)
    prediction = knnAlgorithm.predict(X_test)
    t2 = datetime.now()
    delta = (t2 - t1).total_seconds()
    return prediction, delta

# Hep Data-set
data_sets = [
    {'name': "hepatitis", 'dummy_value': "?", 'class_field': "Class"}
    # ,    {'name': "pen-based", 'dummy_value': "", 'class_field': "a17"}
]
results = pd.DataFrame(columns=['dataset', 'fold', 'dist_metric', 'k_value', 'accuracy', 'true_p_n', 'false_p_n'])
for dataset in data_sets:
    print(dataset)
    for f in range(0, 10):
        print(f)
        fold_path = 'datasets/{0}/{0}.fold.00000{1}'.format(dataset['name'], f)
        X_train, y_train = load_and_normalise(fold_path + '.train.arff', dataset['class_field'], dataset['dummy_value'])
        X_test, y_test = load_and_normalise(fold_path + '.test.arff', dataset['class_field'], dataset['dummy_value'])

        for dist in ['euclidean', 'cosine', 'hamming', 'minkowski', 'correlation']:
            print(dist)
            for k in [1, 3, 5, 7]:
                print(k)
                y_pred, delta = run_knn(kNNAlgorithm(k, metric=dist, p=4), X_train, y_train, X_test)
                knn_accuracy = accuracy_score(y_test, y_pred)
                knn_correct = accuracy_score(y_test, y_pred, normalize=False)
                knn_incorrect = X_test.shape[0] - knn_correct
                results = results.append({
                    'dataset': dataset['name'],
                    'fold': str(f),
                    'dist_metric': dist,
                    'k_value': str(k),
                    'accuracy': str(knn_accuracy),
                    'true_p_n': str(knn_correct),
                    'false_p_n': str(knn_incorrect)
                }, ignore_index=True)
