import matplotlib.pyplot as plt
import numpy as np


def plot_svm_hyperplane(X, y, support_vectors, vector_idx, num_vectors, coefs, intercept, plot_alldata=True):
    # plot x0 and x1, support vectors and hyperplane

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_svm_vectors(X, y, support_vectors, vector_idx, num_vectors, fig, ax, plot_alldata)

    if type(coefs[0]) is np.ndarray:
        # coefs are the coeficents of the hyperplane. Coefs is the w vector in  w * x + b = 0
        # building hyperplane line:
        xx = np.linspace(X[:, 0].min(), X[:, 0].max())
        yy = (-coefs[:, 0] * xx - (intercept[:]))
        yy = yy / coefs[:, 1]
        m, b = np.polyfit(xx, yy, 1)
        ax.plot(xx, m * xx + b, '-')

    plt.show(block=True)

    return fig, ax



def plot_svm_vectors(X, y, support_vectors, vector_idx, num_vectors, fig=None, ax=None, plot_alldata=True):
    i = 0
    colors = ['b', 'g', 'y', 'r', 'orange']
    classes = np.unique(y)

    show = False

    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        show = True

    if plot_alldata:
        for cl in classes:
            ax.scatter(X[y == cl, 0], X[y == cl, 1], s=10, edgecolors=colors[i % len(colors)],
                       facecolors='none', linewidths=2, label=('Class ' + str(cl)))
            i += 1

    i = 0
    j = 0
    colors = ['r', 'orange', 'b', 'g', 'y', ]
    for vec in support_vectors:

        if i >= num_vectors[j]:
            j += 1
            i = 0

        ax.scatter(vec[0], vec[1], s=20, edgecolors=colors[j % len(colors)],
                   facecolors='none', linewidths=2, label=('SV class ' + str(classes[j])))
        i += 1

    if show:
        plt.show(block=True)

    return fig, ax


def plot_test_data(X_test, y_predict, y_true, fig=None, ax=None):

    i = 0
    colors = ['b', 'greenyellow', 'y', 'r', 'orange']
    classes = np.unique([y_true, y_predict])

    show = False

    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        show = True

    for cl in classes:
        ax.scatter(X_test[y_true == cl, 0], X_test[y_true == cl, 1], s=10, marker=".", edgecolors=colors[i % len(colors)],
                   facecolors='none', linewidths=2, label=('Class ' + str(cl)))
        i += 1

    i = 0

    colors = ['b', 'greenyellow', 'orange', 'g', 'y', ]
    for cl in classes:
        ax.scatter(X_test[y_predict == cl, 0], X_test[y_predict == cl, 1], s=40, marker="s", edgecolors=colors[i % len(colors)],
                   facecolors='none', linewidths=2, label=('Class ' + str(cl)))
        i += 1


    if show:
        plt.show(block=True)

    return fig, ax


def emerrf_plot_svm(clf, X, y, X_test=None, y_test=None, y_predict=None,
                    filename=None, title=None):
    # Based on:
    # http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html

    plt.scatter(X[:, 0], X[:, 1], c=y, marker='.', s=30, cmap=plt.cm.Set1, alpha=0.5)
    legend_labels = ["Train data"]

    is_test_data = X_test is not None and y_test is not None
    if is_test_data:
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='v', s=30,
                    cmap=plt.cm.Set1)
        legend_labels.append("Test data")

    if is_test_data and y_predict is not None:
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_predict, marker='^', s=30,
                    cmap=plt.cm.Set1)
        legend_labels.append("Prediction")

    plt.legend(legend_labels)
    if title:
        plt.title(title)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=100,
               linewidth=1, facecolors='none')

    if filename:
        print("Saving figure: {}".format(filename))
        plt.savefig(filename, dpi=150)

    plt.show(block=True)
