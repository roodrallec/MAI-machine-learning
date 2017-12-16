# coding=utf-8
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
from scipy.stats import friedmanchisquare
from nemenyi import kw_nemenyi
from kNNAlgorithm import *
from Parser import *
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime


def train_test_split(path, class_field, dummy_nominal):
    matrix, labels, class_, _ = read_dataset(path, class_field, dummy_nominal)
    return matrix, class_


def norm_train_test_split(path, class_field, dummy_value):
    X_train, y_train = train_test_split(path + '.train.arff', class_field, dummy_value)
    X_test, y_test = train_test_split(path + '.test.arff', class_field, dummy_value)
    X_train, _min, _max = normalize_min_max(X_train)
    X_test, _min, _max = normalize_min_max(X_test)
    return X_train, y_train, X_test, y_test


def run_knn(knnAlgorithm, X_train, y_train, X_test):
    t1 = datetime.now()
    knnAlgorithm.fit(X_train, y_train)
    prediction = knnAlgorithm.predict(X_test)
    t2 = datetime.now()
    delta = (t2 - t1).total_seconds()
    return prediction, delta


data_sets = [{'name': "hepatitis", 'dummy_value': "?", 'class_field': "Class"}]
# ,{'name': "pen-based", 'dummy_value': "", 'class_field': "a17"}]
dist_metrics = ['euclidean', 'cosine', 'hamming', 'minkowski', 'correlation']
k_values = [1, 3, 5, 7]
results = pd.DataFrame(columns=['dataset', 'fold', 'dist_metric', 'k_value', 'run_time', 'c_matrix', 'accuracy'])

for dataset in data_sets:

    for f in range(0, 10):
        path = 'datasets/{0}/{0}.fold.00000{1}'.format(dataset['name'], f)
        X_train, y_train, X_test, y_test = norm_train_test_split(path, dataset['class_field'], dataset['dummy_value'])

        for dist in dist_metrics:

            for k in k_values:
                algo = kNNAlgorithm(k, metric=dist, p=4, policy='voting', weights=None, selection=None)
                y_pred, delta = run_knn(algo, X_train, y_train, X_test)
                c_matrix = confusion_matrix(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                results = results.append({'dataset': dataset['name'], 'fold': f, 'dist_metric': dist, 'k_value': k,
                                          'run_time': delta, 'c_matrix': c_matrix, 'accuracy': acc}, ignore_index=True)
                print(dataset['name'], f, dist, k, 'c_matrix' + str(c_matrix))

"""
    FRIEDMAN TESTS
    
    The Friedman test (Friedman, 1937, 1940) is a non-parametric equivalent of the repeated-measures ANOVA. 
    It ranks the algorithms for each data set separately, the best performing algorithm getting the rank of 1, 
    the second best rank 2. . . . In case of ties (like in iris, lung cancer, mushroom and primary 
    tumor), average ranks are assigned.
"""
alpha = 0.1
sorted_results = results.sort_values(['dataset', 'dist_metric', 'k_value'], ascending=[1,1,1])
grouped_accuracies = np.array_split(sorted_results['accuracy'], len(data_sets)*len(k_values)*len(dist_metrics))
friedman_chi, p_value = friedmanchisquare(*grouped_accuracies)

if (p_value > alpha):
    print("Accept null hypothesis", "p=" + str(p_value), "alpha=" + str(alpha))
else:
    print("Rejecting null hypothesis")
    H, p_omnibus, p_corrected, reject = kw_nemenyi(grouped_accuracies)
    print("Nemenyi scores", H, p_omnibus)
