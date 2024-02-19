import os
import pickle
import sys
from os.path import exists
import numpy as np
from dtaidistance import dtw, dtw_ndim
from cobras_ts.cobras_dtw import COBRAS_DTW
from cobras_ts.querier.labelquerier import LabelQuerier
import matplotlib.pyplot as plt
import datetime as dt
from queue import Queue
from sklearn.metrics.cluster import adjusted_rand_score
from multi_explain_util import ExplanationBaseMutli

from matplotlib.patches import ConnectionPatch

datatypelist = ['_TRAIN', '_TEST']
class ExplainCobrasMulti(ExplanationBaseMutli):
    def __init__(self, series, x, max_iter, k, ml, cl, clustering, dtw_ndim_type, window):
        self.clustering = clustering
        super(ExplainCobrasMulti, self).__init__(series, x, max_iter, k)
        self.cluster_sps_dict = self.get_cluster_sequences_dict()
        self.current_ml_dict = self.list_to_dic(self.find_links_from_current_representatives(ml))
        self.current_cl_dict = self.list_to_dic(self.find_links_from_current_representatives(cl))
        self.ml_dict = self.list_to_dic(ml)
        self.cl_dict = self.list_to_dic(cl)
        self.dtw_ndim_type = dtw_ndim_type
        self.window = window


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
        found_must_link_chain_between_superinstances, must_link_chain = self.outer_layer_bfs(first_repre_idx,
                                                                                             second_repre_id)
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

    def find_cannot_link_chain(self, cluster_idx_left, cluster_idx_right, first_repre_idx, second_repre_idx, first,
                               second):
        cannot_link_between_cluster = self.get_cannot_link_between_cluster(cluster_idx_left,
                                                                                                       cluster_idx_right)
        k = 1
        for cannot_lin in cannot_link_between_cluster:
            found_first, must_link_first = self.outer_layer_bfs(first_repre_idx, cannot_lin[0])
            found_second, must_link_second = self.outer_layer_bfs(second_repre_idx, cannot_lin[1])
            if found_first & found_second:
                # if k > current_cannot_link_length:
                #     print("find CANNOT link in older generation pool")

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

    # randomly pick one pair of cannot link
    def get_cannot_link_between_cluster(self, cluster_idx_left, cluster_idx_right):
        left_cluster_sps_indices = []
        right_cluster_sps_indices = []
        for sp in self.clustering.clusters[cluster_idx_left].super_instances:
            left_cluster_sps_indices.append(sp.representative_idx)
        for sp in self.clustering.clusters[cluster_idx_right].super_instances:
            right_cluster_sps_indices.append(sp.representative_idx)

        cannot_link_pairs_of_current = []
        # cannot_link_pairs_of_old_generation = []
        for sp_left in left_cluster_sps_indices:
            if sp_left in self.current_cl_dict:
                for sp_right in right_cluster_sps_indices:
                    if sp_right in self.current_cl_dict[sp_left]:
                        cannot_link_pairs_of_current.append([sp_left, sp_right])
                    # elif sp_right in self.cl_dict[sp_left]:
                    #     cannot_link_pairs_of_old_generation.append([sp_left, sp_right])

            #     for sp_right in right_cluster_sps_indices:
            #         if sp_right in self.cl_dict[sp_left]:
            #             cannot_link_pairs_of_old_generation.append([sp_left, sp_right])

        return cannot_link_pairs_of_current
               # + cannot_link_pairs_of_old_generation, len(cannot_link_pairs_of_current)


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
            print("there are "+ str(len(superintances)) + " in cluster " +str(i))
            for instance in superintances:
                representatives.append(instance.representative_idx)
                print("cluster " + str(i) + ", representative is " + str(instance.representative_idx) + " and contains " + str(len(instance.indices)) + " indices " + str(
                    instance.indices))
            cluster_sps_dict[i] = representatives
        return cluster_sps_dict

    def get_y_label(self, instanceid, clusterid):
        return "iid: " + str(instanceid) + "\ncid: " + str(clusterid)

    def plot_musk_link_chain(self, cluster_id, idx_left, idx_right, repre_idx_left, repre_idx_right):
        must_link, is_first_normal, is_second_normal = \
            self.find_must_link_chain(idx_left, idx_right, repre_idx_left, repre_idx_right)

        fig, axs = plt.subplots(nrows=len(must_link), ncols=self.dim, sharex=True, sharey=True, squeeze=False)

        # fig.suptitle('two TS in the same cluster')
        for plot_idx in range(0, len(must_link)):
            if self.dim == 1:
                axs[plot_idx, 0].plot(self.x, self.series[must_link[plot_idx]])
                axs[plot_idx, 0].set_ylabel(self.get_y_label(must_link[plot_idx], cluster_id), color="blue",
                                                      rotation=0)
            else:
                for current_dim in range(0, self.dim):
                    axs[plot_idx,current_dim].plot(self.x,
                                                    self.get_single_ts(self.series, must_link[plot_idx], current_dim)       )
                                                   # self.series[must_link[plot_idx]][current_dim])
                    # if current_dim == 0:
                        # axs[plot_idx,current_dim].set_ylabel(self.get_y_label(must_link[plot_idx], cluster_id), color="blue", rotation=0, labelpad=10)
                    # if current_dim == self.dim - 1 and plot_idx > 0: #last column and not the first row
                    #     if not (plot_idx == 1 and is_first_normal or plot_idx == len(must_link) - 1 and is_second_normal):
                    #         axs[plot_idx, current_dim].text(
                    #            'MUST LINK',
                    #             color = 'green',
                    #             horizontalalignment='right',
                    #             verticalalignment='bottom',
                    #             transform=axs[plot_idx, current_dim].transAxes
                    #             )
        self.plot_alignment_info(must_link, is_first_normal, is_second_normal, axs)
        print("ml chain " + str(must_link))

    def plot_cannot_link_chain(self, cluster_id_1, cluster_id_2, repre_idx_left, repre_idx_right, first, second):
        cannot_link_1, cannot_link_2, is_first_normal, is_second_normal = self.find_cannot_link_chain(
            cluster_id_1, cluster_id_2, repre_idx_left, repre_idx_right, first, second
        )

        fig2, axs2 = plt.subplots(nrows=len(cannot_link_1) + len(cannot_link_2), ncols=self.dim, sharex=True, sharey=True, squeeze=False)
        # fig2.suptitle('two TS in different clusters')

        for plot_idx in range(0, len(cannot_link_1)):
            if self.dim == 1:
                axs2[plot_idx, 0].plot(self.x, self.series[cannot_link_1[plot_idx]],
                                                 color='orange')
                axs2[plot_idx, 0].set_ylabel(
                    self.get_y_label(cannot_link_1[plot_idx], cluster_id_1), color="orange", rotation=0
                )
            else:

                for current_dim in range(0, self.dim):
                    axs2[plot_idx, current_dim].plot(self.x,
                                                    self.get_single_ts(self.series, cannot_link_1[plot_idx], current_dim),
                                                     # self.series[cannot_link_1[plot_idx]][current_dim],
                                                     color='orange')
                    # if current_dim == 0:
                    #     axs2[plot_idx, current_dim].set_ylabel(
                    #     self.get_y_label(cannot_link_1[plot_idx], cluster_id_1), color="orange", rotation=0,labelpad=10
                    # )
            # axs2[plot_idx].tick_params(axis='y', labelrotation=90)
        for plot_idx in range(0, len(cannot_link_2)):
            if self.dim == 1:
                axs2[len(cannot_link_1) + plot_idx, 0].plot(self.series[cannot_link_2[plot_idx]],
                                                                      color='purple')
                axs2[len(cannot_link_1) + plot_idx, 0].set_ylabel(
                    self.get_y_label(cannot_link_2[plot_idx], cluster_id_2), color="purple", rotation=0
                )
            else:

                for current_dim in range(0, self.dim):
                    axs2[len(cannot_link_1) + plot_idx, current_dim].plot(self.get_single_ts(self.series, cannot_link_2[plot_idx], current_dim),
                        # self.series[cannot_link_2[plot_idx]][current_dim],
                                                                          color='purple')
                    # if current_dim == 0:
                    #     axs2[len(cannot_link_1) + plot_idx, current_dim].set_ylabel(
                    #     self.get_y_label(cannot_link_2[plot_idx], cluster_id_2), color="purple", rotation=0, labelpad=10
                    # )
        self.plot_alignment_info(cannot_link_1 + cannot_link_2, is_first_normal, is_second_normal, axs2)
        print("cl chain " + str(cannot_link_1) + "-cl-" +str(cannot_link_2))

    def plot_alignment_info(self, link_chain, is_first_normal, is_second_normal, axs):
        if is_first_normal:
            self.find_alignment_info_between_two_timeseries(self.series[link_chain[0]], self.series[link_chain[1]],axs, 0, 1)
        if is_second_normal:
            len_must_link_chain = len(link_chain)
            self.find_alignment_info_between_two_timeseries(
                self.series[link_chain[len_must_link_chain - 2]], self.series[link_chain[len_must_link_chain - 1]], axs, len_must_link_chain - 2, len_must_link_chain - 1)

    def find_alignment_info_between_two_timeseries(self, seq1, seq2, axs, rowid1, rowid2):
        if self.dim == 1:
            path = dtw.warping_path(seq1, seq2)
            for (idx1, idx2) in path:
                cp = ConnectionPatch(
                    (idx1, seq1[idx1]), (idx2, seq2[idx2]), "data", "data",
                    axesA=axs[rowid1, 0], axesB=axs[rowid2, 0], linestyle="--", color="blue"
                )
                axs[rowid2, 0].add_artist(cp)
        else:
            if self.dtw_ndim_type == 'd':
                path = dtw_ndim.warping_path(seq1, seq2, window = int(self.window * 0.01 * self.length))
                for (idx1, idx2) in path:
                    for colid in range(0, self.dim):
                        cp = ConnectionPatch(
                            (idx1, seq1[idx1, colid]), (idx2, seq2[idx2, colid]), "data", "data",
                            axesA=axs[rowid1, colid], axesB=axs[rowid2, colid], linestyle="--", color="blue", linewidth=0.01
                            )
                        axs[rowid2, colid].add_artist(cp)

            else:
                for colid in range(0, self.dim):
                    path = dtw.warping_path(seq1[:,colid], seq2[:,colid], window = int(self.window * 0.01 * self.length))
                    for (idx1, idx2) in path:
                        cp = ConnectionPatch(
                            (idx1, seq1[idx1, colid]), (idx2,  seq2[idx2, colid]), "data", "data",
                            axesA=axs[rowid1, colid], axesB=axs[rowid2, colid], linestyle="--", color="blue", linewidth=0.01
                        )
                        axs[rowid2, colid].add_artist(cp)

    def plot_constraint_link_chain(self):
        idx1 = int(input("index1"))
        idx2 = int(input("index2"))
        cluster_idx_left, cluster_idx_right, repre_idx_left, repre_idx_right = self.get_cluster_and_representative_ids(
            self.clustering, idx1, idx2
        )
        if cluster_idx_left == cluster_idx_right:
            self.plot_musk_link_chain(cluster_idx_left, idx1, idx2, repre_idx_left, repre_idx_right)
        else:
            self.plot_cannot_link_chain(cluster_idx_left, cluster_idx_right, repre_idx_left, repre_idx_right, idx1,
                                        idx2)

    def plot_specific_dots_around_a_centroid(self):
        idx = int(input("centroid index"))

    def get_predicted_labels_from_cobras_result(self, clustering, series_n):
        predicted_labels = np.zeros(series_n)
        cluster_n = len(clustering.clusters)
        for i in range(cluster_n):
            predicted_labels[self.find_all_indices_of_one_cluster(i)] = i
        return predicted_labels

def prepare_affinity_matrix(data_cnt, ucr_path, datasources, affinity_matrix_file_path, datatypeid, alpha, window, dtw_ndim_type='i'):
        # load the datas

    datatype = datatypelist[datatypeid]
    load_data_from_file = lambda x: np.loadtxt(os.path.join(ucr_path, x, x + datatype), delimiter=',')[:data_cnt, 1:]
    data = list(map(load_data_from_file, datasources))

    labels = np.loadtxt(os.path.join(ucr_path, datasources[0], datasources[0] + datatype), delimiter=',')[:data_cnt, 0]

    length_of_sequence = data[0].shape[1]
    number_of_sequence = data[0].shape[0]
    dimension = len(datasources)
    print("series shape = " + str(length_of_sequence))
    print("series number = " + str(number_of_sequence))

    if exists(affinity_matrix_file_path):  # no need to contruct the affinity matrix
        print("affinity matrix file " + affinity_matrix_file_path +" already exists ...")
        fileread = open(affinity_matrix_file_path, 'rb')
        affinities = pickle.load(fileread)
        fileread.close()
    else:
        # construct the affinity matrix
        print("creating a new affinity matrix file " + affinity_matrix_file_path)
        dt1 = dt.datetime.now()
        if dtw_ndim_type == 'd':  #dtw dependent
            series = np.zeros((number_of_sequence, length_of_sequence, dimension))
            for id_of_sequence in np.arange(number_of_sequence):
                for id_of_dim in np.arange(dimension):
                    to_be_added = data[id_of_dim][id_of_sequence]
                    series[id_of_sequence, :, id_of_dim] = to_be_added
            dists = dtw_ndim.distance_matrix(
                series, parallel=True, use_mp=True,
                window=int(0.01 * window * length_of_sequence)
               )

        else: # dtw independent
            to_distance_matrix = lambda x: dtw.distance_matrix(
                x, parallel=True, use_mp=True,
                window=int(0.01 * window * length_of_sequence)
                )
            distance_matrix_list = list(map(to_distance_matrix, data))
            dists = distance_matrix_list[0] + distance_matrix_list[1] + distance_matrix_list[2]

        dt2 = dt.datetime.now()
        print("distance matrix use time " + str((dt2 - dt1).seconds))
        dists[dists == np.inf] = 0
        dists = dists + dists.T - np.diag(np.diag(dists))
        affinities = np.exp(-dists * alpha)
        filewrite = open(affinity_matrix_file_path, 'wb')
        pickle.dump(affinities, filewrite)
        filewrite.close()
    return affinities, data, labels
def explain_multi_dimension(affinities, data, labels, budget, dimension, clustering_result_path,dtw_ndim_type):
    iteration = 1
    length_of_sequence = data[0].shape[1]
    number_of_sequence = data[0].shape[0]
    print("series shape = " + str(length_of_sequence))
    print("series number = " + str(number_of_sequence))
    print("query budget is " + str(budget))

    start = 0
    end = length_of_sequence

    # initialise cobras_dtw with the precomputed affinities
    if exists(clustering_result_path):
        print("clustering result already exists in " + clustering_result_path)
        with open(clustering_result_path, "rb") as f:
            clustering = pickle.load(f)
            ml = pickle.load(f)
            cl = pickle.load(f)
            f.close()
    else:
        dt0 = dt.datetime.now()
        print("creating new clustering result in " + clustering_result_path)
        clusterer = COBRAS_DTW(affinities, LabelQuerier(labels), budget)
        clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()
        print("clustering result prepared ")
        dt1  =dt.datetime.now()
        print("building clustering result using " + str((dt1 - dt0).seconds))

        with open(clustering_result_path, "wb") as f:
            pickle.dump(clustering, f)
            pickle.dump(ml, f)
            pickle.dump(cl, f)
            f.close()

    print("there are " + str(len(clustering.clusters)) + " clusters")

    #build new multi-dimension series
    series = np.zeros((number_of_sequence, length_of_sequence, dimension))
    for id_of_sequence in np.arange(number_of_sequence):
        for id_of_dim in np.arange(dimension):
            to_be_added = data[id_of_dim][id_of_sequence]
            series[id_of_sequence, :, id_of_dim] = to_be_added

    explain_cobras = ExplainCobrasMulti(series, (start, end), iteration, len(clustering.clusters), ml, cl, clustering, dtw_ndim_type, window)
    predicted_labels = explain_cobras.get_predicted_labels_from_cobras_result(clustering, number_of_sequence)

    ari = adjusted_rand_score(labels_true=labels, labels_pred=predicted_labels)
    print("the adjust rand score for cobras is: " + str(ari))
    explain_cobras.explain_result(len(clustering.clusters))

if __name__ == '__main__':
    ucr_path = sys.argv[1]
    datasources = sys.argv[2].split(",")
    datatypeid = int(sys.argv[3]) # 0 for train, 1 for test
    data_cnt = int(sys.argv[4])
    dataset = sys.argv[5]
    dtw_ndim_type = sys.argv[9]

    dataset_basic_name = dataset +datatypelist[datatypeid]+"_"+ str(data_cnt)+"_" + dtw_ndim_type
    affinity_matrix_folder =  sys.argv[6]
    affinity_matrix_file_path = affinity_matrix_folder + dataset_basic_name + ".pkl"
    alpha = 0.5
    window = 10
    affinity_matrix, data, labels, = prepare_affinity_matrix(data_cnt, ucr_path, datasources, affinity_matrix_file_path, datatypeid, alpha, window, dtw_ndim_type)


    clustering_reuslt_folder =  sys.argv[7]
    budget = int(sys.argv[8])
    clustering_result_path = clustering_reuslt_folder + dataset + "/" + dataset_basic_name + "_budget_" + str(budget) + ".pkl"
    explain_multi_dimension(affinity_matrix, data, labels, budget, len(datasources), clustering_result_path, dtw_ndim_type)

    #a configuration example
    # / Users / ary / Desktop / UCR_TS_Archive_2015
    # "uWaveGestureLibrary_X,uWaveGestureLibrary_Y,uWaveGestureLibrary_Z"
    # 0
    # 896
    # uWaveGestureLibrary
    # / Users / ary / PycharmProjects / cobras / examples / affinities_permanent /
    # / Users / ary / PycharmProjects / cobras / examples / clustering_result /
    # 50
    # d

    #for command line

    #remote
    #ssh simiao@pinac37.cs.kuleuven.be
    # python3 multi_explain_cobras_dtw.py /home/simiao/ucr_ts "uWaveGestureLibrary_X,uWaveGestureLibrary_Y,uWaveGestureLibrary_Z" 1 3582 uWaveGestureLibrary /home/simiao/afm/ /home/simiao/cres/ 50 d

    #local
    # python3 multi_explain_cobras_dtw.py /Users/ary/Desktop/UCR_TS_Archive_2015 "uWaveGestureLibrary_X,uWaveGestureLibrary_Y,uWaveGestureLibrary_Z" 0 30 uWaveGestureLibrary /Users/ary/PycharmProjects/cobras/examples/affinities_permanent/ /Users/ary/PycharmProjects/cobras/examples/clustering_result/ 50 d
