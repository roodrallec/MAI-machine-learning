# coding=utf-8
"""
    KNN performance analysis
    A Python function that automatically repeats the process for the 10-fold cross-validation files. That is,
    it reads automatically each training case and runs each one of the test cases in a selected classifier.

    For evaluating the performance of the KNN algorithm, we will use the percentage of correctly classified
    instances. The average accuracy over the 10-fold cross-validation sets is calculated with friedmann acceptance test.
"""
from knn_utils import *
from kNNAlgorithm import *
from Parser import *
# DEFAULT VALUES
LOAD_PICKLE = False
SAVE_PICKLE = True
NULL_ACCEPT = 0.1
DEFAULT_K = [1, 3, 5, 7]
DEFAULT_DIST = ['euclidean', 'cosine', 'hamming', 'minkowski', 'correlation']
DEFAULT_A_PARAMS = [{'name': "Weighted knn", 'sel_method': 'relief', 'num_features': [0]},
                    {'name': "Selection knn", 'sel_method': 'information_gain', 'num_features': [0]}]


def main_run(data_sets, k_values=DEFAULT_K, dist_metrics=DEFAULT_DIST, algo_params=DEFAULT_A_PARAMS):
    results = pd.DataFrame(columns=['algorithm', 'dataset', 'fold', 'dist_metric', 'k_value', 'run_time', 'c_matrix',
                                    'heaviest_feature', 'accuracy'])
    for dataset in data_sets:
        for f in range(0, 10):
            path = 'datasets/{0}/{0}.fold.00000{1}'.format(dataset['name'], f)
            X_train, y_train, X_test, y_test = norm_train_test_split(path, dataset['class_field'], dataset['dummy_value'])

            for params in algo_params:
                for sf in params['num_features']:
                    weights, w_idx = knn_weights(X_train, y_train, sf, **params)
                    print('IDXs of weights, heaviest desc', w_idx)

                    for dist in dist_metrics:
                        for k in k_values:
                            algorithm = kNNAlgorithm(k, metric=dist, p=4, policy='voting', weights=weights)
                            delta, c_matrix, accuracy = run_knn(algorithm, X_train, y_train, X_test, y_test)
                            results = results.append({'algorithm': params['name'], 'num_features': np.int(sf),
                                                      'dataset': dataset['name'], 'fold': f, 'dist_metric': dist,
                                                      'k_value': k, 'run_time': delta, 'c_matrix': c_matrix,
                                                      'heaviest_feature': w_idx[0], 'accuracy': accuracy},
                                                     ignore_index=True)
                            print(params['name'], dataset['name'], f, dist, k, 'c_matrix' + str(c_matrix), accuracy)
    return results


def acceptance_test(results, accept=NULL_ACCEPT, folds=10):
    accuracies = [list(results[results['fold'] == fold]['accuracy']) for fold in range(0, folds)]
    accuracies = np.transpose(np.array(accuracies))
    return friedman_test(list(accuracies), accept)


# Load results from file if LOAD_PICKLE flag is True
hep_res_part1 = pd.read_pickle("hep_res_part1.df") if LOAD_PICKLE else None
hep_res_part2 = pd.read_pickle("hep_res_part2.df") if LOAD_PICKLE else None
penb_res_part1 = pd.read_pickle("penb_res_part1.df") if LOAD_PICKLE else None
penb_res_part2 = pd.read_pickle("penb_res_part2.df") if LOAD_PICKLE else None
""" 
    Hepatitis Part I:
    We select Euclidean with K = 7 (Highest Median with lowest IQR (interquartile range))
"""
hepa_data_set = [{'name': "hepatitis", 'dummy_value': "?", 'class_field': "Class"}]
no_feature_algo = [{'name': "plain knn", 'sel_method': 'None', 'num_features': [0]}]

if hep_res_part1 is None:
    hep_res_part1 = main_run(hepa_data_set, algo_params=no_feature_algo)

if SAVE_PICKLE:
    hep_res_part1.to_pickle("hep_res_part1.df")

w3plot(hep_res_part1, part=1, filename="hepa_res_part1.png")
accept, p_value, mean_ranks, p_values = acceptance_test(hep_res_part1)
print('ACCEPT:', accept, 'MEAN_RANKS', mean_ranks, 'P_VALUES', p_value)
"""
    Hepatitis Part II:
"""
hepa_algo_params = [{'name': "Weighted knn", 'sel_method': 'relief', 'num_features': [0]},
                    {'name': "Selection knn", 'sel_method': 'information_gain', 'num_features': range(1, 20),
                     'discrete_features': 'auto'}]

if hep_res_part2 is None:
    hep_res_part2 = main_run(hepa_data_set, k_values=[7], dist_metrics=['euclidean'], algo_params=hepa_algo_params)

if SAVE_PICKLE:
    hep_res_part2.to_pickle("hep_res_part2.df")

w3plot(hep_res_part2, part=2, filename="hepa_res_part2.png")
accept, p_value, mean_ranks, p_values = acceptance_test(hep_res_part2)
print('ACCEPT:', accept, 'MEAN_RANKS', mean_ranks, 'P_VALUES', p_value)
# """
#     Pen-based Part I:
#     Euclidian k = 3, all algos have the same results. Standard distance selected with best K median.
# """
# penb_data_set = [{'name': "pen-based", 'dummy_value': "", 'class_field': "a17"}]
#
# if penb_res_part1 is None:
#     penb_res_part1 = main_run(penb_data_set, dist_metrics=['euclidean', 'cosine', 'hamming', 'minkowski'],
#                               algo_params=no_feature_algo)
# if SAVE_PICKLE:
#     penb_res_part1.to_pickle("penb_res_part1.df")
#
# w3plot(penb_res_part1, part=1, filename="penb_res_part1.png")
# w3plot(
#     penb_res_part1[~penb_res_part1["dist_metric"].isin(["hamming"])], part=1, filename="penb_res_part1_no_hamming.png"
# )
# # Discard Hamming distance as low performant
# penb_res_part1 = penb_res_part1[~penb_res_part1["dist_metric"].isin(["hamming"])]
# accept, p_value, mean_ranks, p_values = acceptance_test(penb_res_part1)
# # No point of applying Nemenyi test, p-value of 0.06 and absolute differences are between 0.98-0.99 of accurancy
# print('ACCEPT:', accept, 'MEAN_RANKS', mean_ranks, 'P_VALUES', p_value)
# """
#     Pen-based Part II:
# """
# penb_algo_params = [{'name': "Weighted knn", 'sel_method': 'relieff', 'num_features': [0]},
#                     {'name': "Selection knn", 'sel_method': 'information_gain', 'num_features': range(1, 16),
#                      'discrete_features': False}]
#
# if penb_res_part2 is None:
#     penb_res_part2 = main_run(penb_data_set, k_values=[3], dist_metrics=['euclidean'], algo_params=penb_algo_params)
#
# if SAVE_PICKLE:
#     penb_res_part2.to_pickle("penb_res_part2.df")
#
# w3plot(penb_res_part2, part=2, filename="penb_res_part2.png")
# # Conclusions:
# # - the more features the better accuracy in Selection
# #   a) we consider that the pen-based was made by experts considering all features are important for the classification
# #   b) chart confirm the trend of increasing accuracy by adding features
# # - Compare the weighted against the best selection candidate
# #   weighted candidate vs against last/best (two) selection candidate
# #   results are no differences p-value of 0.246, as all seems to be important, weighted should be fine
# penb_res_part2_2 = penb_res_part2[(penb_res_part2["algorithm"] == "Weighted knn") | (penb_res_part2["num_features"] >= 17.0)]
# accept, p_value, mean_ranks, p_values = acceptance_test(penb_res_part2_2)
# print('ACCEPT:', accept, 'MEAN_RANKS', mean_ranks, 'P_VALUES', p_value)