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
import random
from sklearn.metrics.cluster import adjusted_rand_score
from tsbuble.utils import ExplanationBase
from matplotlib.patches import ConnectionPatch


class ExplainCobras(ExplanationBase):
    def __init__(self, series, x, max_iter, k, ml, cl, clustering):
        self.clustering = clustering
        super(ExplainCobras, self).__init__(series, x, max_iter, k)
        self.cluster_sps_dict = self.get_cluster_sequences_dict()
        self.current_ml_dict = self.list_to_dic(self.find_links_from_current_representatives(ml))
        self.current_cl_dict = self.list_to_dic(self.find_links_from_current_representatives(cl))
        self.ml_dict = self.list_to_dic(ml)
        self.cl_dict = self.list_to_dic(cl)


    def find_links_from_current_representatives(self, link_pool):
        current_representatives = []
        for representatives in self.cluster_sps_dict.values():
            current_representatives += representatives
        current_repre_link_pool = []
        for link in link_pool:
            if link[0] in current_representatives and link[1] in current_representatives:
                current_repre_link_pool.append(link)
        return current_repre_link_pool

    def find_all_indices_of_one_cluster(self, cluster_idx):
        superinstances = self.clustering.clusters[cluster_idx].super_instances
        indices_of_current_cluster = []
        for superinstance in superinstances:
            indices_of_current_cluster += superinstance.indices
        return indices_of_current_cluster

    ## non directed graph
    def list_to_dic(self, link_pool):
        dict = {}
        for link in link_pool:
            if link[0] in dict:
                dict[link[0]].append(link[1])
            else:
                dict[link[0]] = [link[1]]
            if link[1] in dict:
                dict[link[1]].append(link[0])
            else:
                dict[link[1]] = [link[0]]
        return dict

    def get_cluster_and_representative_ids(self, clustering, firstId, secondId):
        found_first_sp = False
        found_second_sp = False
        representative_idx_first = 0
        representative_idx_second = 0
        cluster_idx_first = 0
        cluster_idx_second = 0

        for cluster_idx in range(0, len(clustering.clusters)):
            cluster = clustering.clusters[cluster_idx]
            for sp in cluster.super_instances:
                if found_first_sp == True and found_second_sp == True:
                    break
                for idx in sp.indices:
                    if found_first_sp == False and firstId == idx:
                        cluster_idx_first = cluster_idx
                        representative_idx_first = sp.representative_idx
                        found_first_sp = True
                    if found_second_sp == False and secondId == idx:
                        cluster_idx_second = cluster_idx
                        representative_idx_second = sp.representative_idx
                        found_second_sp = True
        # assert (found_first_sp == True and found_second_sp == True)
        return cluster_idx_first, cluster_idx_second, representative_idx_first, representative_idx_second

    def find_must_link_chain(self, first_idx, second_idx, first_repre_idx, second_repre_id):
        found_must_link_chain_between_superinstances, must_link_chain = self.outer_layer_bfs(first_repre_idx, second_repre_id)
        assert (found_must_link_chain_between_superinstances == True)
        result_link = must_link_chain
        is_first_normal = False
        is_second_normal = False

        if must_link_chain[0] != first_idx:
            result_link = [first_idx] + result_link
            is_first_normal = True
        if must_link_chain[len(must_link_chain) - 1] != second_idx:
            result_link = result_link + [second_idx]
            is_second_normal = True
        return result_link, is_first_normal, is_second_normal


    def find_cannot_link_chain(self, cluster_idx_left, cluster_idx_right, first_repre_idx, second_repre_idx, first, second):
        cannot_link_between_cluster, current_cannot_link_length = self.get_cannot_link_between_cluster(cluster_idx_left, cluster_idx_right)
        k = 1
        for cannot_lin in cannot_link_between_cluster:
            found_first, must_link_first = self.outer_layer_bfs(first_repre_idx, cannot_lin[0])
            found_second, must_link_second = self.outer_layer_bfs(second_repre_idx, cannot_lin[1])
            if found_first & found_second:
                if k > current_cannot_link_length:
                    print("find CANNOT link in older generation pool")

                is_first_normal = False
                is_second_normal = False
                if must_link_first[0] != first:
                    must_link_first = [first] + must_link_first
                    is_first_normal = True
                if must_link_second[len(must_link_second) - 1] != second:
                    must_link_second = must_link_second + [second]
                    is_second_normal = True
                break
            k += 1

        return must_link_first, must_link_second, is_first_normal, is_second_normal

    #randomly pick one pair of cannot link
    def get_cannot_link_between_cluster(self, cluster_idx_left, cluster_idx_right):
        left_cluster_sps_indices = []
        right_cluster_sps_indices = []
        for sp in self.clustering.clusters[cluster_idx_left].super_instances:
            left_cluster_sps_indices.append(sp.representative_idx)
        for sp in self.clustering.clusters[cluster_idx_right].super_instances:
            right_cluster_sps_indices.append(sp.representative_idx)

        cannot_link_pairs_of_current = []
        cannot_link_pairs_of_old_generation = []
        for sp_left in left_cluster_sps_indices:
            if sp_left in self.current_cl_dict:
                for sp_right in right_cluster_sps_indices:
                    if sp_right in self.current_cl_dict[sp_left]:
                        cannot_link_pairs_of_current.append([sp_left, sp_right])
                    elif sp_right in self.cl_dict[sp_left]:
                        cannot_link_pairs_of_old_generation.append([sp_left, sp_right])

            elif sp_left in self.cl_dict:
                for sp_right in right_cluster_sps_indices:
                    if sp_right in self.cl_dict[sp_left]:
                        cannot_link_pairs_of_old_generation.append([sp_left, sp_right])

        return cannot_link_pairs_of_current + cannot_link_pairs_of_old_generation, len(cannot_link_pairs_of_current)


    # return <find_chain_or_not, the found path>
    def outer_layer_bfs(self, start, end):
        if start == end:
            return True, [start]
        else:
            is_path_found_among_current_representatives, path_found = self.bfs(start, end, self.current_ml_dict)
            if is_path_found_among_current_representatives:
                return True, path_found
            else:
                print("find MUST link in older generation pool")
                return self.bfs(start, end, self.ml_dict)

    def bfs(self, start, end, link_pool):
        visited_vertex = []
        visited_path = Queue()
        visited_path.put([start])
        visited_vertex.append(start)

        while not visited_path.empty():
            current_path = visited_path.get()
            last_of_path = current_path[len(current_path) - 1]

            if last_of_path in link_pool:
                for next_elem in link_pool[last_of_path]:
                    if next_elem not in visited_vertex:  # make sure there is no loop
                        to_be_added_path = current_path + [next_elem]
                        if next_elem == end:
                            return True, to_be_added_path
                        else:
                            visited_vertex.append(next_elem)
                            visited_path.put(to_be_added_path)

        return False, None

    def get_cluster_sequences_dict(self):
        cluster_sps_dict = {}
        for i in range(0, len(self.clustering.clusters)):
            representatives = []
            superintances = self.clustering.clusters[i].super_instances
            for instance in superintances:
                representatives.append(instance.representative_idx)
                print("cluster " + str(i) + ", representative " + str(instance.representative_idx) + " indices " + str(instance.indices))
            cluster_sps_dict[i] = representatives
            print(str(i) + " : " + str(len(superintances)))
        return cluster_sps_dict

    def get_y_label(self, instanceid, clusterid):
        return "iid: " + str(instanceid) + "\ncid: " + str(clusterid)

    def plot_musk_link_chain(self, cluster_id, idx_left, idx_right, repre_idx_left, repre_idx_right):
        must_link, is_first_normal, is_second_normal = \
            self.find_must_link_chain(idx_left, idx_right, repre_idx_left, repre_idx_right)

        fig, axs = plt.subplots(len(must_link), sharex=True, sharey=True)

        fig.suptitle('two TS in the same cluster')
        for plot_idx in range(0, len(must_link)):
            axs[plot_idx].plot(self.x, self.series[must_link[plot_idx]])
            axs[plot_idx].set_ylabel(self.get_y_label(must_link[plot_idx], cluster_id), color="blue", rotation=0,  labelpad=10)
        self.plot_alignment_info(must_link, is_first_normal, is_second_normal, axs)

    def plot_cannot_link_chain(self, cluster_id_1, cluster_id_2, repre_idx_left, repre_idx_right, first, second):
        cannot_link_1, cannot_link_2, is_first_normal, is_second_normal = self.find_cannot_link_chain(
            cluster_id_1, cluster_id_2, repre_idx_left, repre_idx_right, first, second
            )


        fig2, axs2 = plt.subplots(len(cannot_link_1) + len(cannot_link_2), sharex=True, sharey=True)

        fig2.suptitle('two TS in different clusters')


        for plot_idx in range(0, len(cannot_link_1)):
            axs2[plot_idx].plot(self.x, self.series[cannot_link_1[plot_idx]], color='orange')
            axs2[plot_idx].set_ylabel(
                self.get_y_label(cannot_link_1[plot_idx], cluster_id_1), color="orange", rotation=0,  labelpad=10
                )
            # axs2[plot_idx].tick_params(axis='y', labelrotation=90)
        for plot_idx in range(0, len(cannot_link_2)):
            axs2[len(cannot_link_1) + plot_idx].plot(self.series[cannot_link_2[plot_idx]], color='purple')
            axs2[len(cannot_link_1) + plot_idx].set_ylabel(
                self.get_y_label(cannot_link_2[plot_idx], cluster_id_2), color="purple", rotation=0,  labelpad=10
                )
        self.plot_alignment_info(cannot_link_1 + cannot_link_2, is_first_normal, is_second_normal, axs2)
    def plot_alignment_info (self, link_chain, is_first_normal, is_second_normal, axs):
        if is_first_normal:
            self.find_alignment_info_between_two_timeseries(self.series[link_chain[0]], self.series[link_chain[1]], axs[0], axs[1])
        if is_second_normal:
            len_must_link_chain = len(link_chain)
            self.find_alignment_info_between_two_timeseries(
                self.series[link_chain[len_must_link_chain - 2]], self.series[link_chain[len_must_link_chain - 1]],
                axs[len_must_link_chain - 2], axs[len_must_link_chain - 1]
                )

    def find_alignment_info_between_two_timeseries (self, seq1, seq2, ax1, ax2):
        path = dtw.warping_path(seq1, seq2)
        for (idx1, idx2) in path:
            cp = ConnectionPatch(
                (idx1, seq1[idx1]), (idx2, seq2[idx2]), "data", "data",linewidth=0.01,
                axesA=ax1, axesB=ax2, linestyle="--", color="blue"
                )
            ax2.add_artist(cp)

    def plot_constraint_link_chain(self):
        idx1 = int(input("index1"))
        idx2 = int(input("index2"))
        cluster_idx_left, cluster_idx_right, repre_idx_left, repre_idx_right = self.get_cluster_and_representative_ids(
            self.clustering, idx1, idx2
            )
        if cluster_idx_left == cluster_idx_right:
            self.plot_musk_link_chain(cluster_idx_left, idx1, idx2, repre_idx_left, repre_idx_right)
        else:
            self.plot_cannot_link_chain(cluster_idx_left, cluster_idx_right, repre_idx_left, repre_idx_right, idx1, idx2)

    def plot_specific_dots_around_a_centroid(self):
        idx = int(input("centroid index"))

    def get_predicted_labels_from_cobras_result(self, clustering, series_n):
        predicted_labels = np.zeros(series_n)
        cluster_n = len(clustering.clusters)
        for i in range(cluster_n):
            predicted_labels[self.find_all_indices_of_one_cluster(i)] = i
        return predicted_labels

def explain_sinlge_dimension():
    # configuration
    ucr_path = '/Users/ary/Desktop/UCR_TS_Archive_2015'
    dataset = 'Trace'
    budget = 50
    alpha = 0.5
    window = 10
    iteration = 1
    affinity_matrix_path = "/Users/ary/PycharmProjects/cobras/examples/affinities_permanent/" + dataset + ".pkl"
    clustering_result_path = "/Users/ary/PycharmProjects/cobras/examples/clustering_result/" + dataset +"/" + str(budget) + ".pkl"

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

    start = 0
    end = series.shape[1]
    affinities = None
    if exists(affinity_matrix_path): # no need to contruct the affinity matrix
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

    print("there are " + str(len(clustering.clusters)) + " clusters")
    explain_cobras = ExplainCobras(series, (start, end), iteration, len(clustering.clusters), ml, cl, clustering)
    predicted_labels = explain_cobras.get_predicted_labels_from_cobras_result(clustering, series.shape[0])

    ari = adjusted_rand_score(labels_true=labels, labels_pred=predicted_labels)
    print("the adjust rand score for cobras is: " + str(ari))
    explain_cobras.explain_result(len(clustering.clusters))


if __name__ == '__main__':
    explain_sinlge_dimension()