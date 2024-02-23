import os
import pickle
from os.path import exists
import numpy as np
from dtaidistance import dtw
from cobras_ts.cobras_dtw import COBRAS_DTW
from cobras_ts.querier.labelquerier import LabelQuerier
import matplotlib.pyplot as plt
import datetime as dt
from queue import Queue
from sklearn.metrics.cluster import adjusted_rand_score
from tsbubble.utils import ExplanationBase
from matplotlib.patches import ConnectionPatch






class COBRAS_DTW:
    def find_all_indices_of_one_cluster(self, clustering_result, cluster_idx):
        superinstances = clustering_result[cluster_idx].super_instances
        indices_of_current_cluster = []
        for superinstance in superinstances:
            indices_of_current_cluster += superinstance.indices
        return indices_of_current_cluster

    def datapreprocessing(self):
        ucr_path = '/Users/ary/Desktop/UCR_TS_Archive_2015'
        dataset = 'Trace'
        budget = 50
        alpha = 0.5
        window = 10
        iteration = 1
        affinity_matrix_path = "/Users/ary/PycharmProjects/cobras/examples/affinities_permanent/" + dataset + ".pkl"
        clustering_result_path = "/Users/ary/PycharmProjects/cobras/examples/clustering_result/" + dataset + "/" + str(
            budget) + ".pkl"

        # load the datas
        data = np.loadtxt(os.path.join(ucr_path, dataset, dataset + '_TEST'), delimiter=',')
        data_train = np.loadtxt(os.path.join(ucr_path, dataset, dataset + '_TRAIN'), delimiter=',')
        # data = np.vstack((data, data_train))
        # labels = data[:, 0][:num_of_series_to_test]
        labels = data[:, 0]
        series = data[:, 1:]
        print("series shape = " + str(series.shape[1]))
        print("series number = " + str(series.shape[0]))
        print("dataset is " + dataset + " ... query budget is " + str(budget))

        affinities = None
        if exists(affinity_matrix_path):  # no need to contruct the affinity matrix
            print("affinity matrix already exists ...")
            fileread = open(affinity_matrix_path, 'rb')
            affinities = pickle.load(fileread)
            fileread.close()
        else:
            # construct the affinity matrix
            print("creating a new affinity matrix ...")
            dt1 = dt.datetime.now()
            dists = dtw.distance_matrix(series, parallel=True, use_mp=True, window=int(0.01 * window * series.shape[1]))
            dt2 = dt.datetime.now()
            print("distance matrix use time " + str((dt2 - dt1).seconds))
            dists[dists == np.inf] = 0
            dists = dists + dists.T - np.diag(np.diag(dists))
            affinities = np.exp(-dists * alpha)
            filewrite = open(affinity_matrix_path, 'wb')
            pickle.dump(affinities, filewrite)
            filewrite.close()

        # initialise cobras_dtw with the precomputed affinities
        if exists(clustering_result_path):
            print("clustering result already exists ...")
            with open(clustering_result_path, "rb") as f:
                clustering = pickle.load(f)
                ml = pickle.load(f)
                cl = pickle.load(f)
                f.close()
        else:
            print("creating new clustering result ...")
            clusterer = COBRAS_DTW(affinities, LabelQuerier(labels), budget)
            clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()
            with open(clustering_result_path, "wb") as f:
                pickle.dump(clustering, f)
                pickle.dump(ml, f)
                pickle.dump(cl, f)
                f.close()
        cluster_and_idx = dict()
        print("there are " + str(len(clustering.clusters)) + " clusters")
        for i in np.arange(len(clustering.clusters)):
            cluster_and_idx[i] = set(self.find_all_indices_of_one_cluster(clustering.clusters, i))
        return series, cluster_and_idx

from tsbubble.ts_bubble import TsBubble

if __name__ == '__main__':
    cobras_dtw = COBRAS_DTW()
    series, cluster_and_idx = cobras_dtw.datapreprocessing()
    print(str(series.shape))

    ts_bubble_cobras_dtw = TsBubble()
    alignments = ts_bubble_cobras_dtw.mapping(series, cluster_and_idx)

    for cls_id in np.arange(len(cluster_and_idx)):
        current_alignment_info = alignments[cls_id]
        average = current_alignment_info.series_mean
        assoc_tabs = current_alignment_info.assoc_tab
        assoc_timeaxis_tabs = current_alignment_info.assoc_timeaxis_tab
        cur_series = current_alignment_info.cur_series
        shifts_optimal = ts_bubble_cobras_dtw.find_the_optimal_shifts(assoc_timeaxis_tabs)
        all_time_series = [average] + cur_series
        ts_bubble_cobras_dtw.plot_bubble_of_one_dimension(all_time_series, len(average),
                                                          assoc_tabs, assoc_timeaxis_tabs, len(all_time_series),
                                                          shifts_optimal)