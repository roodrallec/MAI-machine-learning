import pandas as pd
from nemenyi import NemenyiTestPostHoc
from knn_utils import *
import sklearn_relief as relief
import sklearn.feature_selection as skfs
from scipy.stats import friedmanchisquare
from kNNAlgorithm import *
from Parser import *
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime


def norm_train_test_split(path, class_field, dummy_value):
    X_train, labels, y_train, _ = read_dataset(path + '.train.arff', class_field, dummy_value)
    X_test, labels, y_test, _ = read_dataset(path + '.test.arff', class_field, dummy_value)
    X_train, _min, _max = normalize_min_max(X_train)
    X_test, _min, _max = normalize_min_max(X_test)
    return X_train, y_train, X_test, y_test


def run_knn(knnAlgorithm, X_train, y_train, X_test, y_test):
    t1 = datetime.now()
    knnAlgorithm.fit(X_train, y_train)
    y_pred = knnAlgorithm.predict(X_test)
    t2 = datetime.now()
    delta = (t2 - t1).total_seconds()
    c_matrix, accuracy = confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred)
    return delta, c_matrix, accuracy


def knn_weights(X_train, y_train, sf, **kwargs):
    num_features = sf
    weights = None
    if kwargs["sel_method"] == 'information_gain':
        weights = skfs.mutual_info_classif(X_train, y_train, discrete_features=kwargs["discrete_features"])

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


def friedman_test(accuracies_array, alpha=0.1):
    friedman_chi, p_value = friedmanchisquare(*accuracies_array)
    accept = True
    mean_ranks = None
    p_values = None

    if p_value <= alpha:
        accept = False
        nemenyi = NemenyiTestPostHoc(np.asarray(accuracies_array))
        mean_ranks, p_values = nemenyi.do()

    return accept, p_value, mean_ranks, p_values


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