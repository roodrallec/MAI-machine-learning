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
import sklearn_relief as relief
import sklearn.feature_selection as skfs
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


def knn_weights(X_train, y_train, sf, **kwargs):
    num_features = sf
    weights = None
    if kwargs["sel_method"] == 'information_gain':
        weights = skfs.mutual_info_classif(X_train, y_train, discrete_features=kwargs["descrete_features"])

        if num_features > 0:
            w_idx = np.argsort(-weights)
            weights[w_idx[:num_features]] = 1
            weights[w_idx[num_features:]] = 0

    elif kwargs["sel_method"] == 'relief':
        r = relief.Relief(n_features=num_features)
        r.fit_transform(X_train, y_train)
        weights = r.w_

    elif kwargs["sel_method"] == 'relieff':
        r = relief.ReliefF()
        r.fit_transform(X_train, y_train)
        weights = r.w_

    return weights


def w3plot(results, part=1, engine="seaborn", filename=None):
    import matplotlib.pyplot as plt
    if engine == "seaborn":
        import seaborn as sns
        sns.set()
        if part == 1:
            fig, ax = plt.subplots(1, 1, figsize=(14, 7))
            ax = sns.boxplot(x="k_value", y="accuracy", hue="dist_metric",
                             data=results, palette="Set1")
            ax.set_title("Parameters Optimization")
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("K of KNN")
            l = ax.legend()
            l.set_title("Distances")

        elif part == 2:
            fig, (ax_algo1, ax_algo2) = plt.subplots(1, 2, sharey=True,
                                                     figsize=(14, 7))
            sns.set()

            ax_algo1.set_title('Weighted [RELIEF]')
            sns.boxplot(x="k_value", y="accuracy", hue="dist_metric",
                        data=results[
                            results["algorithm"].isin(['Weighted knn'])],
                        palette="Set1", ax=ax_algo1)
            ax_algo1.set_ylabel("Accuracy")
            ax_algo1.set_xlabel("K of KNN")
            ax_algo1_l = ax_algo1.legend()
            ax_algo1_l.set_title("Distances")

            ax_algo2.set_title('Selection [IG]')
            sns.boxplot(x="num_features", y="accuracy", hue="dist_metric",
                        data=results[
                            results["algorithm"].isin(['Selection knn'])],
                        palette="Set1", ax=ax_algo2)
            ax_algo2.set_ylabel("Accuracy")
            ax_algo2.set_xlabel("Num. of Selected Features")
            ax_algo2_l = ax_algo1.legend()
            ax_algo2_l.set_title("Distances")

    else:
        # Pandas
        if part == 1:
            results.boxplot(by=["k_value", "dist_metric"], column=["accuracy"],
                            figsize=(14, 7), vert=False, grid=True)
        elif part == 2:
            results.groupby("algorithm").boxplot(by=["k_value", "dist_metric"],
                                                 column=["accuracy"],
                                                 figsize=(14, 7), vert=False,
                                                 grid=False)
    if filename:
        print("Saving figure: {}".format(filename))
        plt.savefig(filename, dpi=300)


def main_run(data_sets, k_values=None, dist_metrics=None, algo_params=None, friedman=True):

    if not k_values:
        k_values = [1, 3, 5, 7]
    if not dist_metrics:
        dist_metrics = ['euclidean', 'cosine', 'hamming', 'minkowski', 'correlation']
    if not algo_params:
        algo_params = [{
            'name': "Weighted knn",
            'sel_method': 'relief',
            'num_features': [0]
        }, {
            'name': "Selection knn",
            'sel_method': 'information_gain',
            'num_features': [0]
        }]

    results = pd.DataFrame(columns=['algorithm', 'dataset', 'fold', 'dist_metric', 'k_value', 'run_time', 'c_matrix',
                                    'accuracy'])

    for dataset in data_sets:

        for f in range(0, 10):
            path = 'datasets/{0}/{0}.fold.00000{1}'.format(dataset['name'], f)
            X_train, y_train, X_test, y_test = norm_train_test_split(path, dataset['class_field'], dataset['dummy_value'])

            for params in algo_params:
                for sf in params['num_features']:
                    weights = knn_weights(X_train, y_train, sf, **params)

                    for dist in dist_metrics:

                        for k in k_values:
                            y_pred, delta = run_knn(kNNAlgorithm(k, metric=dist, p=4, policy='voting', weights=weights),
                                                    X_train, y_train, X_test)
                            # Confusion matrix and accuracy
                            c_matrix, accuracy = confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred)
                            results = results.append({'algorithm': params['name'], 'num_features': np.int(sf), 'dataset': dataset['name'], 'fold': f,
                                                      'dist_metric': dist, 'k_value': k, 'run_time': delta,
                                                      'c_matrix': c_matrix, 'accuracy': accuracy}, ignore_index=True)
                            print(params['name'], dataset['name'], f, dist, k, 'c_matrix' + str(c_matrix), accuracy)
    if friedman:
        """
            FRIEDMAN TESTS
            
            The Friedman test (Friedman, 1937, 1940) is a non-parametric equivalent of the repeated-measures ANOVA. 
            It ranks the algorithms for each data set separately, the best performing algorithm getting the rank of 1, 
            the second best rank 2. . . . In case of ties (like in iris, lung cancer, mushroom and primary 
            tumor), average ranks are assigned.
        """
        alpha = 0.1
        sorted_results = results.sort_values(['algorithm', 'dataset', 'dist_metric', 'k_value'], ascending=[1, 1, 1, 1])
        grouped_accuracies = np.array_split(sorted_results['accuracy'], len(data_sets)*len(k_values)*len(dist_metrics))
        friedman_chi, p_value = friedmanchisquare(*grouped_accuracies)

        print(results.groupby(['algorithm', 'dataset', 'dist_metric', 'k_value']).mean())

        if (p_value > alpha):
            print("Accept null hypothesis", "p=" + str(p_value), "alpha=" + str(alpha))
        else:
            print("Rejecting null hypothesis")
            H, p_omnibus, p_corrected, reject = kw_nemenyi(grouped_accuracies)
            print("Nemenyi scores", H, p_omnibus)

    return results


hepa_data_set = [{'name': "hepatitis", 'dummy_value': "?", 'class_field': "Class"}]
penb_data_set = [{'name': "pen-based", 'dummy_value': "", 'class_field': "a17"}]

no_feature_algo = [{
    'name': "plain knn",
    'sel_method': 'None',
    'num_features': [0]
}]

# Hepatitis Part I
hep_res_part1 = main_run(hepa_data_set, algo_params=no_feature_algo)
w3plot(hep_res_part1, part=1, filename="hepa_res_part1.png")
# We select Eucleadian with K = 7 (Highest Median with lowest IQR (interquartile range))
# Hepatitis Part II
hepa_algo_params = [{
    'name': "Weighted knn",
    'sel_method': 'relief',
    'num_features': [0]
}, {
    'name': "Selection knn",
    'sel_method': 'information_gain',
    'num_features': range(1, 19),
    'descrete_features': 'auto'
}]

hep_res_part2 = main_run(hepa_data_set, k_values=[7], dist_metrics=['euclidean'], algo_params=hepa_algo_params, friedman=False)
w3plot(hep_res_part2, part=2, filename="hepa_res_part2.png")


## Pen-based Part I
# penb_res_part1 = main_run(penb_data_set, dist_metrics=['euclidean', 'cosine', 'hamming', 'minkowski'], algo_params=no_feature_algo)
# penb_res_part1.to_pickle("penb_res_part1.df")
penb_res_part1 = pd.read_pickle("penb_res_part1.df")
w3plot(penb_res_part1, part=1, filename="penb_res_part1.png")
w3plot(penb_res_part1[~penb_res_part1["dist_metric"].isin(["hamming"])], part=1, filename="penb_res_part1_nohamming.png")
# Euclidian k = 3, all algos have the same results. Standard distance selected with best K median.

# Pen-based Part II
penb_algo_params = [{
    'name': "Weighted knn",
    'sel_method': 'relieff',
    'num_features': [0]
}, {
    'name': "Selection knn",
    'sel_method': 'information_gain',
    'num_features': range(1, 19),
    'descrete_features': False
}]

# penb_res_part2 = main_run(penb_data_set, k_values=[3], dist_metrics=['euclidean'], algo_params=penb_algo_params, friedman=False)
# penb_res_part2.to_pickle("penb_res_part2.df")
penb_res_part2 = pd.read_pickle("penb_res_part2.df")
w3plot(penb_res_part2, part=2, filename="penb_res_part2.png")

