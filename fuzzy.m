data = load('fcmdata.dat');  % load some sample data
n_clusters = 3;              % number of clusters
rng(1)
%[center,U,obj_fcn] = fcm(data, n_clusters);
[center,U,obj_fcn] = fcm(data(:,1), n_clusters);


x = 2