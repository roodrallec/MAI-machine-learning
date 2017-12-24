import matplotlib.pyplot as plt
import numpy as np


def plot_svm_hyperplane(X, y, support_vectors, vector_idx, num_vectors, coefs, intercept):
    # plot x0 and x1, support vectors and hyperplane

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_svm_vectors(X, y, support_vectors, vector_idx, num_vectors, fig, ax)

    # coefs are the coeficents of the hyperplane. Coefs is the w vector in  w * x + b = 0
    # building hyperplane line:
    xx = np.linspace(X[:, 0].min(), X[:, 0].max())
    yy = (-coefs[:, 0] * xx - (intercept[:])) / coefs[:, 1]

    m, b = np.polyfit(xx, yy, 1)
    ax.plot(xx, m * xx + b, '-')

    plt.show(0)

    return fig, ax
    #


def plot_svm_vectors(X, y, support_vectors, vector_idx, num_vectors, fig=None, ax=None):
    i = 0
    colors = ['b', 'g', 'y', 'r', 'orange']
    classes = np.unique(y)

    show = False

    if fig is None or ax is None:
        fig, ax = plt.subplots()
        show = True

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
        plt.show()

    return fig, ax


def plot_test_data(X_test, y_predict, y_true, fig=None, ax=None):

    i = 0
    colors = ['r', 'greenyellow', 'b', 'r', 'orange']
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

    colors = ['r', 'greenyellow', 'orange', 'g', 'y', ]
    for cl in classes:
        ax.scatter(X_test[y_predict == cl, 0], X_test[y_predict == cl, 1], s=40, marker="s", edgecolors=colors[i % len(colors)],
                   facecolors='none', linewidths=2, label=('Class ' + str(cl)))
        i += 1

    print("hold")

    if show:
        plt.show()

    return fig, ax