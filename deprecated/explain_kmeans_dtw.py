import os

import numpy as np
from dtaidistance.clustering.kmeans import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from .old_uni_util import ExplanationBase
from dtaidistance import dtw
from examples.general.mst import mst
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class ExplainKmeans(ExplanationBase):
    def __init__(self, series, span, max_iter, k, dist_matrix, V, cluster_and_idx):
        super().__init__(series, span, max_iter, k)
        self.dist_matrix = dist_matrix
        self.V = V
        self.cluster_and_idx  = cluster_and_idx
        # self.predicted_labels = self.get_predicted_label_from_kmeans_result(cluster_and_idx, V, k)
        # print(self.predicted_labels)
        self.color_arr = cm.rainbow(np.linspace(0, 1, k))

    def get_cluster_sequences_dict(self):
        cluster_sps_dict = {}
        for i in range(0, k):
            size_of_sps_in_current_cluster = len(cluster_and_idx[i])
            cluster_sps_dict[i] = size_of_sps_in_current_cluster
            print(str(i) + " : " + str(size_of_sps_in_current_cluster))
        return cluster_sps_dict

    def find_all_indices_of_one_cluster(self, cluster_idx):
        return cluster_and_idx[cluster_idx]

    def get_predicted_label_from_kmeans_result(self, cluster_and_idx, series_n, k):
        predict_labels = np.zeros(series_n, dtype=int)
        for i in range(k):
            predict_labels[np.array(list(cluster_and_idx[i]))] = i
        return predict_labels

    def plot_constraint_link_chain(self):

        idx1 = int(input("index1"))
        idx2 = int(input("index2"))
        result_chain, dist_chain = mst(idx1, idx2, self.dist_matrix, self.V)
        fig2, axs2 = plt.subplots(len(result_chain), sharex=True, sharey=True)
        fig2.suptitle('similarity chain')

        for plot_idx in range(0, len(result_chain)):
            series_id = result_chain[plot_idx]
            cid = self.predicted_labels[series_id]
            axs2[plot_idx].plot(self.x, series[series_id], color=self.color_arr[cid])
            ylabel = "id: "+str(series_id)+ "\ncid: " + str(cid) if plot_idx == len(dist_chain) else "id: "+str(series_id)+ "\ncid: " + str(cid) + "\ndist: " + str(dist_chain[plot_idx])[:3]
            ylabel = "\ndist: " + str(dist_chain[plot_idx])[:3] if plot_idx < len(dist_chain) else None
            axs2[plot_idx].set_ylabel(ylabel, color= self.color_arr[cid], rotation=0, labelpad=15)
            # axs2[plot_idx].tick_params(axis='y', labelrotation=90)
if __name__ == '__main__':
    # configuration
    ucr_path = '/Users/ary/Desktop/UCR_TS_Archive_2015'
    dataset = 'CBF'
    window = 1
    num_of_series_to_test = 20

    k = 3
    max_it = 50  # max iteration of kmeans
    max_dba_it = 1  # max iteration of dba

    # load the data
    data = np.loadtxt(os.path.join(ucr_path, dataset, dataset + '_TEST'), delimiter=',')
    data_train = np.loadtxt(os.path.join(ucr_path, dataset, dataset + '_TRAIN'), delimiter=',')
    data = np.vstack((data, data_train))

    labels = data[:, 0][:num_of_series_to_test]
    series = data[:, 1:][:num_of_series_to_test]
    print("series shape = " + str(series.shape[1]))
    dists = dtw.distance_matrix(series, window=int(0.1 * window * series.shape[1]))

    start = 0
    end = series.shape[1]

    kmeans = KMeans(
        k=k, max_it=max_it, max_dba_it=max_dba_it, drop_stddev=1,
        nb_prob_samples=0,
        dists_options={"window": window},
        initialize_with_kmedoids=False,
        initialize_with_kmeanspp=True
        )
    cluster_and_idx, performed_it = kmeans.fit(series, use_c=False, use_parallel=False)
    result_chain, dist_chain = mst(4,5, dists, num_of_series_to_test)
    print(result_chain, dist_chain)
    explain_kmeans = ExplainKmeans(series, (start, end), max_dba_it, k, dist_matrix=dists, V=num_of_series_to_test, cluster_and_idx=cluster_and_idx)
    # predicted_labels = explain_kmeans.get_predicted_label_from_kmeans_result(cluster_and_idx, num_of_series_to_test, k)
    adjusted_rand_score = adjusted_rand_score(labels_true=labels, labels_pred=explain_kmeans.predicted_labels)
    print("the adjust rand score for kmeans is: " + str(adjusted_rand_score))
    explain_kmeans.explain_result(k)
