# coding=utf-8
"""
    KNN performance analysis
    A Python function that automatically repeats the process for the 10-fold cross-validation files. That is,
    it reads automatically each training case and runs each one of the test cases in a selected classifier.

    For evaluating the performance of the KNN algorithm, we will use the percentage of correctly classified
    instances. The average accuracy over the 10-fold cross-validation sets is calculated with friedmann acceptance test.
"""
from utils import *
from sklearn.svm import SVC
from Parser import *
# DEFAULT VALUES
LOAD_PICKLE = False
SAVE_PICKLE = True
NULL_ACCEPT = 0.1
DEFAULT_DEGREE = [1, 2, 3, 4, 5]
DEFAULT_MAX_ITER = [-1,1000]
DEFAULT_DECISION_F = ['ovo','ovr']
DEFAULT_KERNEL = ['linear', 'rbf', 'sigmoid']
DEFAULT_C =  np.arange(0.1,1.1,0.1)


def main_run(data_sets, max_iter=DEFAULT_MAX_ITER, decision_f=DEFAULT_DECISION_F, C=DEFAULT_C, kernel=DEFAULT_KERNEL, plot_fig=False):

    results = pd.DataFrame(columns=['algorithm', 'dataset', 'fold', 'C', 'kernel', 'run_time', 'c_matrix', 'accuracy'])

    for dataset in data_sets:
        for f in range(0, 10):
            path = 'datasets/{0}/{0}.fold.00000{1}'.format(dataset['name'], f)
            X_train, y_train, X_test, y_test = norm_train_test_split(path, dataset['class_field'], dataset['dummy_value'])

            for k in kernel:
                for c in C:
                    #for mi in max_iter:
                        #for df in decision_f:
                            algorithm = SVC(C=c, kernel=k)
                            delta, c_matrix, accuracy = run_svm(algorithm, k, X_train, y_train, X_test, y_test, plot_fig=plot_fig)
                            results = results.append({'kernel': k, 'C': c,
                                                      'dataset': dataset['name'], 'fold': f, 'run_time': delta,
                                                      'c_matrix': c_matrix, 'accuracy': accuracy}, ignore_index=True)
                            print(dataset['name'], f, k, c, 'c_matrix' + str(c_matrix), accuracy)
    return results


def acceptance_test(results, accept=NULL_ACCEPT, folds=10):
    accuracies = [list(results[results['fold'] == fold]['accuracy']) for fold in range(0, folds)]
    accuracies = np.transpose(np.array(accuracies))
    return friedman_test(list(accuracies), accept)


# Load results from file if LOAD_PICKLE flag is True
hep_res_part1 = pd.read_pickle("hep_res_part1.df") if LOAD_PICKLE else None
penb_res_part1 = pd.read_pickle("penb_res_part1.df") if LOAD_PICKLE else None

"""
     Hepatitis Part I:
""" hepa_data_set = [{'name': "hepatitis", 'dummy_value': "?", 'class_field': "Class"}]

if hep_res_part1 is None:
     hep_res_part1 = main_run(hepa_data_set, plot_fig=False)

if SAVE_PICKLE:
     hep_res_part1.to_pickle("hep_res_part1.df")

w3plot(hep_res_part1, part=1, filename="hepa_res_part1.png")
accept, p_value, mean_ranks, p_values = acceptance_test(hep_res_part1)
print('ACCEPT:', accept, 'MEAN_RANKS', mean_ranks, 'P_VALUES', p_value)



"""
    Pen-based Part I:

"""
penb_data_set = [{'name': "pen-based", 'dummy_value': "", 'class_field': "a17"}]

if penb_res_part1 is None:
    penb_res_part1 = main_run(penb_data_set, plot_fig=False)
if SAVE_PICKLE:
    penb_res_part1.to_pickle("penb_res_part1.df")

w3plot(penb_res_part1, part=1, filename="penb_res_part1.png")


accept, p_value, mean_ranks, p_values = acceptance_test(penb_res_part1)
# No point of applying Nemenyi test, p-value of 0.06 and absolute differences are between 0.98-0.99 of accurancy
print('ACCEPT:', accept, 'MEAN_RANKS', mean_ranks, 'P_VALUES', p_value)



