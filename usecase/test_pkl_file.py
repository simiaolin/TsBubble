import pickle
from cobras_ts.clustering import Clustering
from  dtaidistance import dtw

def test_affinities():
    alpha = 0.5
    window = 10
    import numpy as np
    import os
    ucr_path = '/Users/ary/Desktop/UCR_TS_Archive_2015'
    dataset = 'CBF'
    # load the datas
    # data = np.loadtxt(os.path.join(ucr_path, dataset, dataset + '_TEST'), delimiter=',')
    data= np.loadtxt(os.path.join(ucr_path, dataset, dataset + '_TRAIN'), delimiter=',')
    # data = np.vstack((data, data_train))
    # labels = data[:, 0][:num_of_series_to_test]
    labels = data[:, 0]
    series = data[:, 1:]
    print("series shape = " + str(series.shape[1]))
    print("series number = " + str(series.shape[0]))
    import datetime as dt

    # dt1 = dt.datetime.now()
    dists = dtw.distance_matrix(series, window=int(0.01 * window * series.shape[1]))
    # dt2 = dt.datetime.now()
    # print("distance matrix use time " + str((dt2 - dt1).seconds))
    dists[dists == np.inf] = 0
    dists = dists + dists.T - np.diag(np.diag(dists))
    affinities = np.exp(-dists * alpha)
    print(affinities)



def show_clustering_info (clustering: Clustering):
   for cls in clustering.clusters:
       for si in cls.super_instances:
           print("repre " + str(si.representative_idx))
           print("instance " + str(si.indices))

def readPkl():
    path = "/affinities_permanent/clustering_result/CBF/portion_0.05/10.pkl"

    fileread = open(path, 'rb')
    cls_1 = pickle.load(fileread)
    show_clustering_info(cls_1)
    fileread.close()


if __name__ == '__main__':
    readPkl()