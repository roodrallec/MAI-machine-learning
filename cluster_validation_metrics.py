import numpy as np
from scipy.io import arff
from fuzzy import fcm
from algorithms import k_means, Bk_means
from sklearn.metrics import adjusted_rand_score, calinski_harabaz_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Data Import
iris, iris_meta = arff.loadarff("datasets/iris.arff")
iris_data = np.array([iris['sepallength'], iris['sepalwidth'],
                      iris['petallength'], iris['petalwidth']]).transpose()
iris_class = iris['class'].reshape((150, 1))


def compare_cluster_methods(data, labels_true, k_range=range(2, 11)):

    ch_scores_km = []; ch_scores_bkm = []; ch_scores_fcm = []
    ar_scores_km = []; ar_scores_bkm = []; ar_scores_fcm = []
    _, lab_codes = np.unique(labels_true, return_inverse=True)

    for k in k_range:
        _, labels_km = k_means(k, data)
        labels_bkm = Bk_means(data, k)
        _, _, _, labels_fcm = fcm(data, k)

        ch_scores_km.append(calinski_harabaz_score(data, labels_km))
        ch_scores_bkm.append(calinski_harabaz_score(data, labels_bkm))
        ch_scores_fcm.append(calinski_harabaz_score(data, labels_fcm))

        ar_scores_km.append(adjusted_rand_score(lab_codes, labels_km))
        ar_scores_bkm.append(adjusted_rand_score(lab_codes, labels_bkm))
        ar_scores_fcm.append(adjusted_rand_score(lab_codes, labels_fcm))

    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    f.suptitle('Cluster validation metrics')
    ax1.plot(k_range, ch_scores_km, 'r',
             k_range, ch_scores_bkm, 'g',
             k_range, ch_scores_fcm, 'b')
    ax1.set_ylabel('Calinski-Harabaz Index')
    ax1.grid(linestyle='--', linewidth=0.5)
    ax1.legend(handles=[
        mpatches.Patch(color='r', label='K-Means'),
        mpatches.Patch(color='g', label='Bis. K-Means'),
        mpatches.Patch(color='b', label='Fuzzy C-Means')
    ], loc='upper right')

    ax2.plot(k_range, ar_scores_km, 'r',
             k_range, ar_scores_bkm, 'g',
             k_range, ar_scores_fcm, 'b')
    ax2.set_ylabel('Adjusted Rand Score')
    ax2.set_xlabel('Number of clusters')
    ax2.set_xlim(np.min(k_range), np.max(k_range))
    ax2.grid(linestyle='--', linewidth=0.5)

    plt.show()


compare_cluster_methods(iris_data, iris_class)


