import pandas as pd


import plot_svm
from utils import *
from scipy.stats import friedmanchisquare
from Parser import *
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime
from kw_nemenyi import *

def norm_train_test_split(path, class_field, dummy_value):
    X_train, labels, y_train, _ = read_dataset(path + '.train.arff', class_field, dummy_value)
    X_test, labels, y_test, _ = read_dataset(path + '.test.arff', class_field, dummy_value)
    X_train, _min, _max = normalize_min_max(X_train)
    X_test, _min, _max = normalize_min_max(X_test)
    return X_train, y_train, X_test, y_test


def run_svm(svm_model,kernel, X_train, y_train, X_test, y_test, plot_fig=False, ):

    t1 = datetime.now()
    svm_model.fit(X_train, y_train)
    vector_idx = svm_model.support_
    support_vectors = svm_model.support_vectors_
    num_vectors = svm_model.n_support_
    dual_coef = svm_model.dual_coef_
    coef=np.array([0])
    intercept=0
    if kernel is 'linear':
        coef = svm_model.coef_
        intercept = svm_model.intercept_

    # print(coef)
    # print(intercept)
    # print(num_vectors)

    y_pred = svm_model.predict(X_test)
    score = svm_model.score(X_test, y_test)

    t2 = datetime.now()
    delta = (t2 - t1).total_seconds()
    c_matrix, accuracy = confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred)

    if plot_fig:
        # print_performance(y_test, y_predict, score)
        fig, ax = plot_svm.plot_svm_hyperplane(X_train, y_train, support_vectors, vector_idx, num_vectors, coef,
                                               intercept)
        plot_svm.plot_svm_hyperplane(X_test, y_test, support_vectors, vector_idx, num_vectors, coef, intercept)
        plot_svm.plot_test_data(X_test, y_pred, y_test, fig=fig, ax=ax)


    return delta, c_matrix, accuracy


def friedman_test(accuracies_array, alpha=0.1):
    friedman_chi, p_value = friedmanchisquare(*accuracies_array)
    accept = True
    mean_ranks = None
    p_values = None

    if p_value <= alpha:
        accept = False

        # nemenyi = NemenyiTestPostHoc(np.asarray(accuracies_array))
        # mean_ranks, p_values = nemenyi.do()

        H, p_value, p_values, reject = kw_nemenyi(accuracies_array, alpha=alpha)

    return accept, p_value, mean_ranks, p_values


def w3plot(results, part=1, engine="seaborn", filename=None):
    import matplotlib.pyplot as plt
    if engine == "seaborn":
        import seaborn as sns
        sns.set()
        if part == 1:
            fig, ax = plt.subplots(1, 1, figsize=(14, 7))
            ax = sns.boxplot(x="kernel", y="accuracy", hue="C",
                             data=results, palette="Set1")
            ax.set_title("Parameters Optimization")
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Kernel Function")
            l = ax.legend()
            l.set_title("C")

        elif part == 2:
            ig, ax = plt.subplots(1, 1, figsize=(14, 7))
            ax = sns.boxplot(x="max_iter", y="accuracy",
                             data=results, palette="Set1")
            ax.set_title("Parameters Optimization")
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("max iter")

        elif part == 3:
            ig, ax = plt.subplots(1, 1, figsize=(14, 7))
            ax = sns.boxplot(x="decision_f", y="accuracy",
                             data=results, palette="Set1")
            ax.set_title("Parameters Optimization")
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("decision function")



    else:
        # Pandas
        if part == 1:
            results.boxplot(by=["kernel", "C"], column=["accuracy"],
                            figsize=(14, 7), vert=False, grid=True)
        elif part == 2:
            results.groupby("algorithm").boxplot(by=["kernel", "C"],
                                                 column=["accuracy"],
                                                 figsize=(14, 7), vert=False,
                                                 grid=False)
    if filename:
        print("Saving figure: {}".format(filename))
        plt.savefig(filename, dpi=300)