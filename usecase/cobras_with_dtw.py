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




dataset = 'CBF'
portion = 0.1

class COBRAS_WITH_DTW:
    def find_all_indices_of_one_cluster(self, clustering_result, cluster_idx):
        superinstances = clustering_result[cluster_idx].super_instances
        indices_of_current_cluster = []
        for superinstance in superinstances:
            indices_of_current_cluster += superinstance.indices
        return indices_of_current_cluster

    def datapreprocessing(self, budget):
        ucr_path = '/Users/ary/Desktop/UCR_TS_Archive_2015'


        alpha = 0.5
        window = 10

        affinity_matrix_path = "/Users/ary/PycharmProjects/TsBubble/affinities_permanent/affinity_matrix/" + dataset + ".pkl"
        clustering_result_path = "/Users/ary/PycharmProjects/TsBubble/affinities_permanent/clustering_result/" + dataset + "/" + str(
            budget) + ".pkl"

        # load the datas
        data = np.loadtxt(os.path.join(ucr_path, dataset, dataset + '_TEST'), delimiter=',')
        data_train = np.loadtxt(os.path.join(ucr_path, dataset, dataset + '_TRAIN'), delimiter=',')
        data = np.vstack((data, data_train))
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
            clusterer = COBRAS_DTW(affinities, LabelQuerier(labels), budget, dim = 1, series=series, labels=labels, explain_links_after_an_iteration=False)
            clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()
            with open(clustering_result_path, "wb") as f:
                pickle.dump(clustering, f)
                pickle.dump(ml, f)
                pickle.dump(cl, f)
                f.close()
        cluster_and_idx = dict()
        print("there are " + str(len(clustering.clusters)) + " clusters")
        for i in np.arange(len(clustering.clusters)):
            cluster_and_idx[i] = self.find_all_indices_of_one_cluster(clustering.clusters, i)
        return series, cluster_and_idx

from tsbubble.ts_bubble import TsBubble
import sys
def find_lim_across_figure(budget_list):

    lim_for_time_series = [sys.float_info.max, sys.float_info.min]
    for budget in budget_list:

        cobras_dtw = COBRAS_WITH_DTW()
        series, cluster_and_idx = cobras_dtw.datapreprocessing(budget)

        ts_bubble_cobras_dtw = TsBubble()
        # kw = {'penalty':  1}
        alignment_path = "/Users/ary/PycharmProjects/TsBubble/affinities_permanent/alignments/" + dataset + "/portion_" + str(
            portion) + "/" + str(budget) + ".pkl"
        if exists(alignment_path):
            print("alignment (mappings) already exists ...")
            with open(alignment_path, "rb") as f:
                alignments = pickle.load(f)
                f.close()
        else:
            print("creating new alignments ...")
            alignments = ts_bubble_cobras_dtw.mapping(series, cluster_and_idx, max_iter=100)
            with open(alignment_path, "wb") as f:
                pickle.dump(alignments, f)
                f.close()

        for cls_id in np.arange(len(cluster_and_idx)):
            current_alignment_info = alignments[cls_id]
            average = current_alignment_info.series_mean
            assoc_tabs = current_alignment_info.assoc_tab
            assoc_timeaxis_tabs = current_alignment_info.assoc_timeaxis_tab
            cur_series = current_alignment_info.cur_series
            # shifts_optimal = ts_bubble_cobras_dtw.find_the_optimal_shifts(assoc_timeaxis_tabs)
            for ts in cur_series:

                lim_for_time_series[1] = max(lim_for_time_series[1], max(ts))
                lim_for_time_series[0] = min(lim_for_time_series[0], min(ts))

            dtw_vertical_deviation, _ = ts_bubble_cobras_dtw.get_vertical_deviation_and_percent(
                average,
                assoc_tabs)
            for i, timepoint in enumerate(average):
                lim_for_time_series[1] = max(lim_for_time_series[1], timepoint + dtw_vertical_deviation[i])
                lim_for_time_series[0] = min(lim_for_time_series[0], timepoint - dtw_vertical_deviation[i])

    return lim_for_time_series


if __name__ == '__main__':
    budget_list = [10, 30]
    lim_for_time_series = find_lim_across_figure(budget_list)

    for budget in budget_list:
        cobras_dtw = COBRAS_WITH_DTW()
        series, cluster_and_idx = cobras_dtw.datapreprocessing(budget)
        print(str(series.shape))

        ts_bubble_cobras_dtw = TsBubble()
    # kw = {'penalty':  1}
        alignment_path = "/Users/ary/PycharmProjects/TsBubble/affinities_permanent/alignments/" +  dataset + "/portion_" + str(portion) + "/"  + str(budget) + ".pkl"
        if exists(alignment_path):
            print("alignment (mappings) already exists ...")
            with open(alignment_path, "rb") as f:
                alignments = pickle.load(f)
                f.close()
        else:
            print("creating new alignments ...")
            alignments = ts_bubble_cobras_dtw.mapping(series, cluster_and_idx, max_iter=100)
            with open(alignment_path, "wb") as f:
                pickle.dump(alignments, f)
                f.close()

        save_fig_folder_name = '/Users/ary/PycharmProjects/TsBubble/results/cbf_' + str(
            budget) + '_queries_initialization_0.1/'

        for cls_id in np.arange(len(cluster_and_idx)):
            current_alignment_info = alignments[cls_id]
            average = current_alignment_info.series_mean
            assoc_tabs = current_alignment_info.assoc_tab
            assoc_timeaxis_tabs = current_alignment_info.assoc_timeaxis_tab
            cur_series = current_alignment_info.cur_series
            # shifts_optimal = ts_bubble_cobras_dtw.find_the_optimal_shifts(assoc_timeaxis_tabs)
            all_time_series = [average] + cur_series
            shifts_optimal = [np.float64(0)] * len(all_time_series)

            ts_bubble_cobras_dtw.plot_bubble_of_one_dimension(all_time_series, len(average), assoc_tabs,
                                                              assoc_timeaxis_tabs, len(all_time_series), shifts_optimal,
                                                              save_fig_name=save_fig_folder_name + str(cls_id + 1) + ".png", lim_for_time_series=lim_for_time_series)
            # ts_bubble_cobras_dtw.plot_alignment(cur_series, average, assoc_timeaxis_tabs, shifts_optimal)

            # while True:
            #     selection = input("please select to show warping path or point cloud, 'c' for cloud and 'w' for warping ")
            #     try:
            #         if selection == 'c' or selection == "C":
            #             idx2 = int(input("index_id"))
            #             ts_bubble_cobras_dtw.plot_cloud_around_dba(alignments, idx2)
            #         elif selection == 'w' or selection == "W":
            #             id  = int(input("timeseries_id"))
            #             import dtaidistance.dtw_visualisation as dtw_vis
            #             path = dtw.warping_path(cur_series[id], average, **kw)
            #             dtw_vis.plot_warping_single_ax(cur_series[id], average, path)
            #             plt.show()
            #         elif selection == 'b':
            #             break
            #     except Exception as e:
            #         print(e)