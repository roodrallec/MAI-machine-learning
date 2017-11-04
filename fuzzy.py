import numpy as np


def fcm(data, cluster_n, expo=2, max_iter=100, min_impro=1e-5, display=True):
    # %FCM Data set clustering using fuzzy c-means clustering.
    # [CENTER, U, OBJ_FCN] = FCM(DATA, N_CLUSTER) finds N_CLUSTER number of
    # clusters in the data set DATA. DATA is size M-by-N, where M is the number of
    # data points and N is the number of coordinates for each data point. The
    # coordinates for each cluster center are returned in the rows of the matrix
    # CENTER. The membership function matrix U contains the grade of membership of
    # each DATA point in each cluster. The values 0 and 1 indicate no membership
    # and full membership respectively. Grades between 0 and 1 indicate that the
    # data point has partial membership in a cluster. At each iteration, an
    # objective function is minimized to find the best location for the clusters
    # and its values are returned in OBJ_FCN.

    data_n, in_n = data.shape
    obj_fcn = np.zeros((max_iter, 1))

    U = initfcm(cluster_n, data_n)
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
    # STEPFCM One step in fuzzy c-mean clustering.
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
    obj_fcn = np.sum(np.sum(np.power(dist, 2) * mf))

    # calculate new U, suppose expo != 1
    tmp = np.power(dist, -2/(expo-1))
    U_new = tmp/(np.ones((cluster_n, 1))*np.sum(tmp, axis=0, keepdims=True))

    return U_new, center, obj_fcn


def distfcm(center, data):
    # DISTFCM Distance measure in fuzzy c-mean clustering.
    # OUT = DISTFCM(CENTER, DATA) calculates the Euclidean distance
    # between each row in CENTER and each row in DATA, and returns a
    # distance matrix OUT of size M by N, where M and N are row
    # dimensions of CENTER and DATA, respectively, and OUT(I, J) is
    # the distance between CENTER(I,:) and DATA(J,:).
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


def initfcm(cluster_n, data_n):
    # INITFCM Generate initial fuzzy partition matrix for fuzzy c-means clustering.
    # U = INITFCM(CLUSTER-_N, DATA_N) randomly generates a fuzzy partition
    # matrix U that is CLUSTER_N by DATA_N, where CLUSTER_N is number of
    # clusters and DATA_N is number of data points. The summation of each
    # column of the generated U is equal to unity, as required by fuzzy
    # c-means clustering.

    U = np.random.rand(cluster_n, data_n)
    col_sum = np.sum(U, 1)
    U = U/U.sum(axis=0, keepdims=True)

    return U


if __name__ == "__main__":

    # Resoruces
    # https://es.mathworks.com/help/fuzzy/examples/fuzzy-c-means-clustering.html
    # Prepare test data used by Matlab (fcmdata.dat)
    x1 = [0.218959190000000, 0.679296410000000, 0.830965350000000,
          0.0534616350000000, 0.686772710000000, 0.0919648910000000,
          0.653918960000000, 0.701190590000000, 0.910320830000000,
          0.736081880000000, 0.632638570000000, 0.722660400000000,
          0.753355830000000, 0.0726858830000000, 0.272709970000000,
          0.766494780000000, 0.359264980000000, 0.904653090000000,
          0.493976680000000, 0.266144510000000, 0.0737490750000000,
          0.529747390000000, 0.464445820000000, 0.770204550000000,
          0.629543420000000, 0.736224510000000, 0.888572210000000,
          0.513273700000000, 0.591113580000000, 0.537303980000000,
          0.467917370000000, 0.287212370000000, 0.178327700000000,
          0.802405730000000, 0.498480120000000, 0.554583850000000,
          0.890737480000000, 0.624849290000000, 0.714709970000000,
          0.239910800000000, 0.681346210000000, 0.147533000000000,
          0.587186620000000, 0.590108610000000, 0.556146140000000,
          0.408766690000000, 0.564898680000000, 0.488514550000000,
          0.651253740000000, 0.247841760000000, 0.476431800000000,
          0.389314170000000, 0.203250330000000, 0.947486780000000,
          0.131188530000000, 0.885648370000000, 0.0921736300000000,
          0.365339030000000, 0.253057360000000, 0.783153170000000,
          0.349524140000000, 0.215248380000000, 0.679592370000000,
          0.250125590000000, 0.860859840000000, 0.817561480000000,
          0.755843530000000, 0.824697390000000, 0.103433930000000,
          0.576716640000000, 0.876565720000000, 0.440038660000000,
          0.869263740000000, 0.886031120000000, 0.463322730000000,
          0.713422320000000, 0.667679070000000, 0.682049120000000,
          0.315732410000000, 0.467531780000000, 0.319177590000000,
          0.682494230000000, 0.836419880000000, 0.708920610000000,
          0.828707950000000, 0.213546800000000, 0.389853590000000,
          0.776865820000000, 0.783865200000000, 0.282155890000000,
          0.819726090000000, 0.601010110000000, 0.828354720000000,
          0.157731180000000, 0.233599190000000, 0.634717440000000,
          0.794769810000000, 0.696242810000000, 0.752940410000000,
          0.669520640000000, 0.633429910000000, 0.227007720000000,
          0.699834440000000, 0.526123280000000, 0.329666390000000,
          0.485325180000000, 0.860225700000000, 0.556835780000000,
          0.738996610000000, 0.528548270000000, 0.310738870000000,
          0.588119120000000, 0.518083590000000, 0.370226280000000,
          0.475896220000000, 0.0782632080000000, 0.369742010000000,
          0.671784150000000, 0.676236920000000, 0.513936370000000,
          0.728608360000000, 0.720767820000000, 0.321560090000000,
          0.460434170000000, 0.661355500000000, 0.605639700000000,
          0.670098400000000, 0.522807710000000, 0.266613320000000,
          0.246733150000000, 0.817101330000000, 0.160748910000000,
          0.707826300000000, 0.436638450000000, 0.751710090000000,
          0.903301180000000, 0.584787200000000, 0.626861370000000,
          0.659053330000000, 0.538661290000000]
    x2 = [0.719711080000000, 0.313898370000000, 0.479039930000000,
          0.779314850000000, 0.522195680000000, 0.714057610000000,
          0.166317870000000, 0.422224000000000, 0.318835530000000,
          0.457606850000000, 0.143807380000000, 0.320667800000000,
          0.463733230000000, 0.877493110000000, 0.648681260000000,
          0.392388530000000, 0.643075020000000, 0.186951990000000,
          0.548148770000000, 0.736379750000000, 0.790437690000000,
          0.400640690000000, 0.568091720000000, 0.163793440000000,
          0.230562620000000, 0.0659695270000000, 0.230885940000000,
          0.318334220000000, 0.243229820000000, 0.383389570000000,
          0.628534250000000, 0.775198800000000, 0.766308500000000,
          0.255895770000000, 0.338006490000000, 0.515819540000000,
          0.378954360000000, 0.0859750140000000, 0.503418000000000,
          0.670943900000000, 0.0602745970000000, 0.876151230000000,
          0.473662160000000, 0.482292780000000, 0.158423960000000,
          0.672772440000000, 0.423224140000000, 0.241754230000000,
          0.544927610000000, 0.916914290000000, 0.578421350000000,
          0.527620050000000, 0.710233540000000, 0.331179650000000,
          0.756249680000000, 0.288435550000000, 0.736337070000000,
          0.540237230000000, 0.767133020000000, 0.508656920000000,
          0.733880840000000, 0.865960860000000, 0.204144340000000,
          0.828844770000000, 0.393998700000000, 0.490341770000000,
          0.174163480000000, 0.368236730000000, 0.795437080000000,
          0.300309010000000, 0.293538730000000, 0.505369090000000,
          0.288428730000000, 0.138812560000000, 0.555480800000000,
          0.476228260000000, 0.246826230000000, 0.408411800000000,
          0.688394810000000, 0.531925270000000, 0.889980570000000,
          0.481226730000000, 0.143123880000000, 0.483091890000000,
          0.325475270000000, 0.654128260000000, 0.619885400000000,
          0.308873540000000, 0.237656110000000, 0.922675680000000,
          0.168744020000000, 0.276163540000000, 0.509182990000000,
          0.838580580000000, 0.726748890000000, 0.332950710000000,
          0.451263720000000, 0.389262330000000, 0.332034560000000,
          0.504874660000000, 0.428417390000000, 0.848783110000000,
          0.268636180000000, 0.477936540000000, 0.754857670000000,
          0.402831490000000, 0.388889440000000, 0.227969710000000,
          0.486851510000000, 0.303004250000000, 0.592462260000000,
          0.513150490000000, 0.520217140000000, 0.723353040000000,
          0.296630540000000, 0.764597280000000, 0.586505770000000,
          0.216997660000000, 0.0797476350000000, 0.318508880000000,
          0.178822670000000, 0.472587000000000, 0.748577880000000,
          0.348485460000000, 0.0640640590000000, 0.141234070000000,
          0.493180050000000, 0.297528620000000, 0.702935820000000,
          0.661490410000000, 0.514713960000000, 0.885525420000000,
          0.135390500000000, 0.508159260000000, 0.195087630000000,
          0.348705400000000, 0.0805080490000000, 0.477785350000000,
          0.138378220000000, 0.130505930000000]
    fcmdata = np.array([x1, x2]).transpose()

    # from scipy.io.matlab import loadmat
    # U = loadmat('U.mat'); U = U['U']
    #center, U, obj_fcn = fcm(fcmdata[:,0].reshape(140,1), 3)  # 1-D
    center, U, obj_fcn = fcm(fcmdata, 3)

    pass

