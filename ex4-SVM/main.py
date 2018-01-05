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
LOAD_PICKLE = True
SAVE_PICKLE = False
NULL_ACCEPT = 0.05
DEFAULT_KERNEL = ['linear', 'rbf', 'sigmoid', 'poly']
DEFAULT_C = np.array([1, 10, 100, 1000])


def main_run(data_sets, C=DEFAULT_C, kernel=DEFAULT_KERNEL, gammas=['auto'], plot_fig=False):

    results = pd.DataFrame(columns=['dataset', 'fold', 'C', 'kernel', 'run_time', 'c_matrix', 'accuracy'])

    for dataset in data_sets:
        for g in gammas:
            for f in range(0, 10):
                path = 'datasets/{0}/{0}.fold.00000{1}'.format(dataset['name'], f)
                X_train, y_train, X_test, y_test = norm_train_test_split(path, dataset['class_field'], dataset['dummy_value'])

                for k in kernel:
                    for c in C:
                        algorithm = SVC(C=c, kernel=k, gamma=g)
                        delta, c_matrix, accuracy = run_svm(algorithm, k, X_train, y_train, X_test, y_test, plot_fig=plot_fig)
                        results = results.append({'kernel': k, 'C': c, 'gamma': g, 'dataset': dataset['name'], 'fold': f, 'run_time': delta,
                                                  'c_matrix': c_matrix, 'accuracy': accuracy}, ignore_index=True)
                        print(dataset['name'], f, k, c, g, 'c_matrix' + str(c_matrix), accuracy)
    return results


def acceptance_test(results, accept=NULL_ACCEPT, part=1):
    if part==1:
        results2 = results[["kernel", "C", "fold", "accuracy"]]
        results2 = results2.set_index(["kernel", "C"])
    else:
        results2 = results[["gamma", "C", "fold", "accuracy"]]
        results2 = results2.set_index(["gamma", "C", ])

    results2 = results2.pivot(columns="fold")
    labels = ["{}-{}".format(*label) for label in results2.index.tolist()]

    return friedman_test(results2.as_matrix(), labels, accept)

# Load results from file if LOAD_PICKLE flag is True
hep_res_part1 = pd.read_pickle("hep_res_part1.df") if LOAD_PICKLE else None
hep_res_part2 = pd.read_pickle("hep_res_part2.df") if LOAD_PICKLE else None
penb_res_part1 = pd.read_pickle("penb_res_part1.df") if LOAD_PICKLE else None
penb_res_part2 = pd.read_pickle("penb_res_part2.df") if LOAD_PICKLE else None
"""
     Hepatitis Part I:
"""
hepa_data_set = [{'name': "hepatitis", 'dummy_value': "?", 'class_field': "Class"}]
if hep_res_part1 is None:
    hep_res_part1 = main_run(hepa_data_set)

if SAVE_PICKLE:
    hep_res_part1.to_pickle("hep_res_part1.df")

print pd.DataFrame(hep_res_part1.groupby(["kernel", "C"]).accuracy.agg(
    ['mean', 'std'])).transpose().to_csv()

w3plot(hep_res_part1, part=1, filename="hepa_res_part1.png")
accept, p_value, mean_ranks, p_values = acceptance_test(hep_res_part1, part=1)
print('ACCEPT:', accept, 'P_VALUES', p_value)

print("Full Nemenyi p-values Matrix")
print(p_values)
print("Statistically relevant Nemenyi p-values")
print(p_values.stack()[(p_values<NULL_ACCEPT).stack()])

"""
     Hepatitis Part II:
"""
hepa_data_set = [{'name': "hepatitis", 'dummy_value': "?", 'class_field': "Class"}]
if hep_res_part2 is None:
    hep_res_part2 = main_run(hepa_data_set,
                             kernel=['rbf'],
                             C=[1, 5, 10, 15, 20, 30, 40, 50],
                             gammas=np.linspace(0.05, 0.25, 5),
                             plot_fig=False)
if SAVE_PICKLE:
    hep_res_part2.to_pickle("hep_res_part2.df")

print pd.DataFrame(hep_res_part2.groupby(["gamma", "C"]).accuracy.agg(
    ['mean', 'std'])).transpose().to_csv()

w3plot(hep_res_part2, part=2, filename="hepa_res_part2.png")
accept, p_value, mean_ranks, p_values = acceptance_test(hep_res_part2, part=2)
print('ACCEPT:', accept, 'P_VALUES', p_value)

print("Full Nemenyi p-values Matrix")
print(p_values)
print("Statistically relevant Nemenyi p-values")
print(p_values.stack()[(p_values<NULL_ACCEPT).stack()])


"""
    Pen-based Part I:

"""
penb_data_set = [{'name': "pen-based", 'dummy_value': "", 'class_field': "a17"}]

if penb_res_part1 is None:
    penb_res_part1 = main_run(penb_data_set)
if SAVE_PICKLE:
    penb_res_part1.to_pickle("penb_res_part1.df")

print pd.DataFrame(penb_res_part1.groupby(["kernel", "C"]).accuracy.agg(
    ['mean', 'std'])).transpose().to_csv()

w3plot(penb_res_part1, part=1, filename="penb_res_part1.png")
accept, p_value, mean_ranks, p_values = acceptance_test(penb_res_part1, part=1)
# No point of applying Nemenyi test, p-value of 0.06 and absolute differences are between 0.98-0.99 of accurancy
print('ACCEPT:', accept, 'P_VALUES', p_value)

print("Full Nemenyi p-values Matrix")
print(p_values)
print("Statistically relevant Nemenyi p-values")
print(p_values.stack()[(p_values<NULL_ACCEPT).stack()])


"""
    Pen-based Part II:

"""

penb_data_set = [{'name': "pen-based", 'dummy_value': "", 'class_field': "a17"}]

if penb_res_part2 is None:
    penb_res_part2 = main_run(penb_data_set,
                             kernel=['rbf'],
                             C=[50, 75, 100, 125, 250, 500, 750, 1000],
                             gammas=np.linspace(0.05, 0.25, 5), plot_fig=False)
if SAVE_PICKLE:
    penb_res_part2.to_pickle("penb_res_part2.df")

print pd.DataFrame(penb_res_part2.groupby(["gamma", "C"]).accuracy.agg(
    ['mean', 'std'])).transpose().to_csv()

w3plot(penb_res_part2, part=2, filename="penb_res_part2.png")
accept, p_value, mean_ranks, p_values = acceptance_test(penb_res_part2, part=2)
print('ACCEPT:', accept, 'P_VALUES', p_value)

print("Full Nemenyi p-values Matrix")
print(p_values)
print("Statistically relevant Nemenyi p-values")
print(p_values.stack()[(p_values<NULL_ACCEPT).stack()])
