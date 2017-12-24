#############################################################
#############################################################
#############################################################


import numpy as np
# import cvxopt
# import cvxopt.solvers
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import plot_svm
import pandas as pd
import matplotlib.pyplot as plt


def print_performance(y_test, y_predict, score):
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()

    print ("correct prediction: {0}".format(tn + tp))
    print ("incorrect prediction: {0}".format(fn + fp))
    print ("total prediction: {0}".format(tn + fp + fn + tp))
    print ("score: {0}".format(score))


if __name__ == "__main__":
    import pylab as pl


    def generate_data_set1():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def generate_data_set2():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0, 0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def generate_data_set3():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train


    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test


    def run_svm_dataset1():
        X1, y1, X2, y2 = generate_data_set1()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        ####
        # Write here your SVM code and choose a linear kernel
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions

        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        vector_idx = svm_model.support_
        support_vectors = svm_model.support_vectors_
        num_vectors = svm_model.n_support_
        dual_coef = svm_model.dual_coef_
        coef = svm_model.coef_
        intercept = svm_model.intercept_

        #print(coef)
        #print(intercept)

        y_predict = svm_model.predict(X_test)
        score = svm_model.score(X_test, y_test)

        print_performance(y_test, y_predict, score)
        # plot_svm.plot_svm_vectors(X_train, y_train, support_vectors, vector_idx, num_vectors)
        plot_svm.plot_svm_hyperplane(X_train, y_train, support_vectors, vector_idx, num_vectors, coef, intercept)

        plot_svm.plot_svm_hyperplane(X_test, y_test, support_vectors, vector_idx, num_vectors, coef, intercept)

        ####


    def run_svm_dataset2():
        X1, y1, X2, y2 = generate_data_set2()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        #### 
        # Write here your SVM code and choose a linear kernel with the best C parameter
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions

        svm_model = SVC(kernel='rbf')
        svm_model.fit(X_train, y_train)
        vector_idx = svm_model.support_
        support_vectors = svm_model.support_vectors_
        num_vectors = svm_model.n_support_
        dual_coef = svm_model.dual_coef_
        #coef = svm_model.coef_
        #intercept = svm_model.intercept_

        #print(coef)
        #print(intercept)
        print(num_vectors)

        y_predict = svm_model.predict(X_test)
        score = svm_model.score(X_test, y_test)

        print_performance(y_test, y_predict, score)
        plot_svm.plot_svm_vectors(X_train, y_train, support_vectors, vector_idx, num_vectors)
        #plot_svm.plot_svm_hyperplane(X_train, y_train, support_vectors, vector_idx, num_vectors, coef, intercept)

        ####


    def run_svm_dataset3(C=1, plot_fig=True):
        X1, y1, X2, y2 = generate_data_set3()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        #### 
        # Write here your SVM code and use a gaussian kernel 
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions

        svm_model = SVC(C=C, kernel='linear')
        svm_model.fit(X_train, y_train)
        vector_idx = svm_model.support_
        support_vectors = svm_model.support_vectors_
        num_vectors = svm_model.n_support_
        dual_coef = svm_model.dual_coef_
        coef = svm_model.coef_
        intercept = svm_model.intercept_

        # print(coef)
        # print(intercept)
        #print(num_vectors)

        #print(coef)
        #print(intercept)

        y_predict = svm_model.predict(X_test)
        score = svm_model.score(X_test, y_test)

        if plot_fig:
            #print_performance(y_test, y_predict, score)
            #plot_svm.plot_svm_vectors(X_train, y_train, support_vectors, vector_idx, num_vectors)
            fig, ax = plot_svm.plot_svm_hyperplane(X_train, y_train, support_vectors, vector_idx, num_vectors, coef, intercept)
            plot_svm.plot_svm_hyperplane(X_test, y_test, support_vectors, vector_idx, num_vectors, coef, intercept)
            plot_svm.plot_test_data(X_test, y_predict, y_test, fig=fig, ax=ax)



        return C, score
        ####


    #############################################################
    #############################################################
    #############################################################

    # EXECUTE SVM with THIS DATASETS
    run_svm_dataset1()  # data distribution 1
    run_svm_dataset2()   # data distribution 2
    run_svm_dataset3()   # data distribution 3




def iterate_nonlinear_set():

    results = pd.DataFrame(columns=['C', 'score'])

    for c in np.arange(0.1,1,0.1):
        for i in range(0,500):
            Cr,score_r=run_svm_dataset3(C=c,plot_fig=False)
            results = results.append({'C': Cr, 'score': score_r}, ignore_index=True)

    avg_score=results.groupby("C").mean()
    print (avg_score)


#iterate_nonlinear_set()

#############################################################
#############################################################
#############################################################
