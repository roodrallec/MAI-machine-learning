import numpy as np


def fcm(data, cluster_n, expo=2, max_iter=100, min_impro=1e-5, display=True,
        initU=None):
    # Data set clustering using fuzzy c-means clustering.
    # The Function finds N_CLUSTER number of clusters in the data set DATA.
    # DATA is size M-by-N, where M is the number of data points and N is the
    # number of coordinates for each data point. The coordinates for each
    # cluster center are returned in the rows of the matrix CENTER.
    # The membership function matrix U contains the grade of membership of
    # each DATA point in each cluster. The values 0 and 1 indicate no
    # membership and full membership respectively. Grades between 0 and 1
    # indicate that the data point has partial membership in a cluster.
    # At each iteration, an objective function is minimized to find the best
    # location for the clusters and its values are returned in OBJ_FCN.

    data_n, in_n = data.shape
    obj_fcn = np.zeros((max_iter, 1))

    # Perform checks
    assert expo != 1, "Exponential parameter must be different from 1"

    U = initfcm(cluster_n, data_n, initU)
    for i in range(max_iter):
        U, center, obj_fcn[i] = stepfcm(data, U, cluster_n, expo)
        if display:
            print("Iteration count {}, obj. fcn = {}".format(i, obj_fcn[i][0]))
        # check termination condition
        if i > 0:
            if np.abs(obj_fcn[i] - obj_fcn[i-1]) < min_impro:
                break

    obj_fcn = obj_fcn[:(i + 1)]  # Actual number of iterations
    return center, U, obj_fcn


def stepfcm(data, U, cluster_n, expo):
    # One step in fuzzy c-mean clustering.
    # [U_NEW, CENTER, ERR] = STEPFCM(DATA, U, CLUSTER_N, EXPO)
    # performs one iteration of fuzzy c-mean clustering, where
    #
    # DATA: matrix of data to be clustered. (Each row is a data point.)
    # U: partition matrix. (U(i,j) is the MF value of data j in cluster j.)
    # CLUSTER_N: number of clusters.
    # EXPO: exponent (> 1) for the partition matrix.
    # U_NEW: new partition matrix.
    # CENTER: center of clusters. (Each row is a center.)
    # ERR: objective function for partition U.
    #
    # Note that the situation of "singularity" (one of the data points is
    # exactly the same as one of the cluster centers) is not checked.
    # However, it hardly occurs in practice.

    # MF matrix after exponential modification
    mf = np.power(U, expo)

    # New center
    center = (np.dot(mf, data)
              / mf.sum(axis=1, keepdims=True) * np.ones((1, data.shape[1])))

    # fill the distance matrix
    dist = distfcm(center, data)

    # objective function
    obj_fcn = np.sum(np.power(dist, 2) * mf)

    # calculate new U, suppose expo != 1
    tmp = np.power(dist, -2/(expo-1))
    U_new = tmp/(np.ones((cluster_n, 1))*np.sum(tmp, axis=0, keepdims=True))

    return U_new, center, obj_fcn


def distfcm(center, data):
    # Distance measure in fuzzy c-mean clustering.
    # Calculates the Euclidean distance between each row in CENTER and each
    # row in DATA, and returns a distance matrix OUT of size M by N, where
    # M and N are row dimensions of CENTER and DATA, respectively, and
    # OUT(I, J) is the distance between CENTER(I,:) and DATA(J,:).
    out = np.zeros((center.shape[0], data.shape[0]))

    # fill the output matrix
    if center.shape[1] > 1:
        for k in range(center.shape[0]):
            out[k, :] = np.sqrt(np.sum(np.power(
                data - np.ones((data.shape[0], 1)) * center[k, :]
                , 2), axis=1))
    else:  # 1-D data
        for k in range(center.shape[0]):
            out[k, :] = np.abs(center[k]-data).transpose()  # TODO: test

    return out


def initfcm(cluster_n, data_n, initU=None):
    # Generate initial fuzzy partition matrix for fuzzy c-means clustering.
    # The function randomly generates a fuzzy partition matrix U that is
    # CLUSTER_N by DATA_N, where CLUSTER_N is number of clusters and
    # DATA_N is number of data points. The summation of each column of the
    # generated U is equal to unity, as required by fuzzy c-means clustering.

    if initU is None:
        U = np.random.rand(cluster_n, data_n)
    else:
        assert isinstance(initU, np.ndarray), "initU must be a numpy.ndarray"
        assert (cluster_n, data_n) == initU.shape, \
            "The dimensions of initU must be ({},{})".format(cluster_n, data_n)
        U = initU

    U = U/U.sum(axis=0, keepdims=True)
    return U


if __name__ == "__main__":
    pass