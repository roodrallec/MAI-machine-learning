import numpy as np
from scipy.io import arff
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=120)


def plot_eigen_values(eig_values):
    fig, ax = plt.subplots()
    fig.suptitle("Eigen values Plot")
    xvar = np.arange(1, eig_values.shape[0] + 1)

    ax.bar(xvar, eig_values, label="Eigen values")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Eigen value")

    ax2 = plt.twinx()
    ax2.plot(xvar, np.cumsum(eig_values / np.sum(eig_values)), color='red',
             label='Accum. Variability')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("% of variance explained")

    print "Explained variance ratio (comp, %) => {}".format(
        zip(xvar, (eig_values / np.sum(eig_values))))

    fig.legend()
    fig.show()


def plot_components(proj, evec, idx=[0, 1], var_labels=None, scale=True):
    assert len(idx) == 2, "Please select only 2 dimensions/components"

    proj_labels = ["PC{}".format(i+1) for i in idx]
    if not var_labels:
        var_labels = ["VAR{}".format(j + 1) for j in range(evec.shape[0])]

    if scale:
        proj = proj[:, idx]/np.ptp(proj[:, idx], axis=0)

    plt.figure()
    plt.title("PCA Biplot: individuals and variables")
    plt.grid(linestyle='--', linewidth=0.5)
    plt.axhline(0, color='darkgray')
    plt.axvline(0, color='darkgray')

    plt.scatter(proj[:, idx[0]], proj[:, idx[1]], s=2)
    plt.xlabel(proj_labels[0])
    plt.ylabel(proj_labels[1])
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)

    for j, label in enumerate(var_labels):
        plt.arrow(0, 0, evec[j, idx[0]], evec[j, idx[1]], color='r', alpha=0.5)
        plt.text(evec[j, idx[0]] * 1.1, evec[j, idx[1]] * 1.1, label,
                 color='r', ha='center', va='center')

    plt.show()

###

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca = PCA(n_components=2)
# pca.fit(X)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

# # Iris dataset
# iris, iris_meta = arff.loadarff("datasets/iris.arff")
# iris_data = np.array([iris['sepallength'], iris['sepalwidth'],
#                       iris['petallength'], iris['petalwidth']]).transpose()
# iris_class = iris['class'].reshape((150, 1))
#
#
# # PCA Algorithm Steps
# # ROW = individuals
# # COL = variables
# X = iris_data
# col_means = np.mean(X, axis=0)
# X_centered = X - col_means
# X_cov = np.cov(X, rowvar=False)
#
# # Sorted Eigen Values and vectors
# eig_values, eig_vectors = np.linalg.eig(X_cov)
#
# # Plot Eigen values
# plot_eigen_values(eig_values)
#
# # Create projections
# X_proj = np.dot(X_centered, -eig_vectors)
#
# # Plot projections of individuals and variables
# plot_components(X_proj, eig_vectors,
#                 var_labels=['sepallength', 'sepalwidth', 'petallength',
#                             'petalwidth'])


# After some checks of
# Dispatch to the right submethod depending on the chosen solver."
# svd_solver : string {'auto', 'full', 'arpack', 'randomized'}\
# the solver is selected by a default policy based on `X.shape` and
#             `n_components`: if the input data is larger than 500x500 and the
#             number of components to extract is lower than 80% of the smallest
#             dimension of the data, then the more efficient 'randomized'
#             method is enabled. Otherwise the exact full SVD is computed and

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/pca.py#L432

# Wine Dataset
wine, wine_meta = arff.loadarff("datasets/wine.arff")
wine_data = np.array([wine['a1'], wine['a2'], wine['a3'], wine['a4'], wine['a5'],
                     wine['a6'], wine['a7'], wine['a8'], wine['a9'], wine['a10'],
                     wine['a11'], wine['a12'], wine['a13']]).transpose()
wine_class = wine['class'].reshape((178, 1))

wine_data_zs = wine_data.copy()

var_labels = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
              "Total phenols", "Flavanoids", "Nonflavanoid phenols",
              "Proanthocyanins", "Color intensity", "Hue",
              "OD280/OD315 of diluted wines", "Proline"]

# PCA Algorithm Steps
X = wine_data.copy()

# Descriptive statistics
sep = "\t"
print sep.join(["var", "Min.", "1st Qu.", "Median", "Mean", "Std.Dev", "3rd Qu.",
                "Max."])
for i in range(X.shape[1]):
    stats = [
        np.min(X[:, i]),
        np.percentile(X[:, i], 0.25),
        np.percentile(X[:, i], 0.5),
        np.mean(X[:, i]),
        np.std(X[:, i]),
        np.percentile(X[:, i], 0.75),
        np.max(X[:, i])
    ]
    print sep.join([var_labels[i]] + ["%.2f" %s for s in stats])

# Plot 2 original vars,  Magnesium vs Proline
idx = [12, 4]
plt.figure()
plt.title("Scatter plot: {} vs {}".format(var_labels[idx[1]], var_labels[idx[0]]))
plt.grid(linestyle='--', linewidth=0.5)
plt.scatter(wine_data[:, idx[0]], wine_data[:, idx[1]], s=2)
plt.xlabel(var_labels[idx[0]])
plt.ylabel(var_labels[idx[1]])

# Standardize the variables: x_std = (x - mean(x)) / sd(x)
col_means = np.mean(X, axis=0)
col_stdev = np.std(X, axis=0)
X_centered = (X - col_means) / col_stdev

# Correlation Visualization
plt.matshow(np.corrcoef(X_centered.transpose())-np.identity(13),
            cmap="RdBu", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.colorbar()

# Var and Covar matrix
X_cov = np.cov(X, rowvar=False)


# Sorted Eigen Values and vectors
eig_values, eig_vectors = np.linalg.eig(X_cov)

# Plot Eigen values
plot_eigen_values(eig_values)

# Create projections
X_proj = np.dot(X_centered, eig_vectors)

# Plot projections of individuals and variables
plot_components(X_proj, eig_vectors, var_labels=var_labels, scale=False)

