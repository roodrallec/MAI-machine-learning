import numpy as np
from scipy.io import arff
from fuzzy import fcm
from algorithms import k_means, Bk_means
from sklearn.metrics import adjusted_rand_score, calinski_harabaz_score
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def compare_k_m_values(data, labels_true, k_range=range(2, 11),
               m_range=np.linspace(2.0, 10.0, num=5)):

    ch_scores = np.zeros((len(k_range), len(m_range)))
    ar_scores = np.zeros((len(k_range), len(m_range)))
    _, lab_codes = np.unique(labels_true, return_inverse=True)

    for i, k in enumerate(k_range):
        for j, m in enumerate(m_range):
            _, _, _, labels_fcm = fcm(data, k, expo=m, max_iter=1000)
            ch_scores[i, j] = calinski_harabaz_score(data, labels_fcm)
            ar_scores[i, j] = adjusted_rand_score(lab_codes, labels_fcm)

    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    f.suptitle('Fuzzy C-Means validation metrics: k vs m coefs.')
    ax1.plot(k_range, ch_scores)
    ax1.legend(m_range, title="coef. m")
    ax1.set_ylabel('Calinski-Harabaz Index')
    ax1.grid(linestyle='--', linewidth=0.5)

    ax2.plot(k_range, ar_scores)
    ax2.set_ylabel('Adjusted Rand Score')
    ax2.set_xlabel('Number of clusters')
    ax2.grid(linestyle='--', linewidth=0.5)

    plt.show()


def compute_ztest(data, k):

    _, labels_km = k_means(k, data)
    labels_bkm = Bk_means(data, k)
    _, _, _, labels_fcm = fcm(data, k, max_iter=1000)

    ml = [("km", labels_km), ("bkm", labels_bkm), ("fcm", labels_fcm)]
    for method, labels in ml:
        for cls in np.unique(labels):
            for j in range(data.shape[1]):
                overall_data = data[:,j]
                cls_data = data[labels == cls,j]

                overall_mean = np.round(np.mean(overall_data), 3)
                overall_std = np.round(np.std(overall_data), 3)

                cls_mean = np.round(np.mean(cls_data), 3)
                cls_std = np.round(np.std(cls_data), 3)

                ttest = ttest_ind(cls_data, overall_data, equal_var=False)
                line = (
                    "Method: {}\tCls: {}\tVar: {}\tallMean: {}\tallStd: {}\t"
                    "clsMean: {}\tclsStd: {}\tttest: {}\tpvalue: {}".format(
                        method, cls, j, overall_mean, overall_std, cls_mean,
                        cls_std, np.round(ttest.statistic, 3),
                        ttest.pvalue
                    ))

                print line


# # Iris dataset
# np.random.seed(1)
# iris, iris_meta = arff.loadarff("datasets/iris.arff")
# iris_data = np.array([iris['sepallength'], iris['sepalwidth'],
#                       iris['petallength'], iris['petalwidth']]).transpose()
# iris_class = iris['class'].reshape((150, 1))
#
# compare_k_m_values(iris_data, iris_class)
# compute_ztest(iris_data, 3)


# # Balance Dataset
# np.random.seed(151423167)
# bal, bal_meta = arff.loadarff("datasets/bal.arff")
# bal_data = np.array([bal['a1'], bal['a2'], bal['a3'], bal['a4']]).transpose()
# bal_class = bal['class'].reshape((625, 1))
#
# compare_k_m_values(bal_data, bal_class)
# compute_ztest(bal_data, 3)

