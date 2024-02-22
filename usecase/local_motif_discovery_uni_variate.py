import os

import numpy as np

from motif_discovery import MotifDiscovery
from tsbubble.ts_bubble import TsBubble
class MotifDiscoveryUnivarite(MotifDiscovery):
    def datapreprocessing(self):

        path_to_series = "../datasets/loco_motif_discovery/ecg-heartbeat-av.csv"
        # path_to_series = "/Users/ary/PycharmProjects/TsBubble/datasets/loco_motif_discovery/ecg-heartbeat-av.csv"
        f = open(path_to_series)
        series = np.array(f.readlines(), dtype=np.double)
        fs = 128  # sampling frequency

        print(series.shape)
        # z-normalize time series
        series = (series - np.mean(series, axis=0)) / np.std(series, axis=0)

        # Parameter rho determines the 'strictness' of the algorithm
        #   - higher -> more strict (more similarity in discovered motif sets)
        #   - lower  -> less strict (less similarity in discovered motif sets)
        rho = 0.6

        # Number of motifs to be found
        nb_motifs = 2

        # Heartbeats last 0.6s - 1s (equivalent to 60-100 bpm)
        l_min = int(0.6 * fs)
        l_max = int(1 * fs)

        # This parameter determines how much the motifs may overlap (intra and inter motif set)
        overlap = 0
        import locomotif.locomotif as locomotif

        if series.ndim == 1:
            series = np.expand_dims(series, axis=1)

        warping = True
        gamma = 1
        sm = locomotif.similarity_matrix_ndim(series, series, gamma, only_triu=False)
        tau = locomotif.estimate_tau_from_am(sm, rho)

        delta_a = -2 * tau
        delta_m = 0.5
        step_sizes = np.array([(1, 1), (2, 1), (1, 2)]) if warping else np.array([(1, 1)])

        lcm = locomotif.LoCoMotif(series=series, gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m, l_min=l_min,
                                  l_max=l_max, step_sizes=step_sizes)
        lcm._sm = sm
        lcm.align()
        lcm.kbest_paths(vwidth=l_min // 2)
        motif_sets = []
        for (candidate, motif_set), _ in lcm.kbest_motif_sets(nb=1, allowed_overlap=overlap, start_mask=None,
                                                              end_mask=None):
            motif_sets.append(motif_set)

        motif_set_length = len(motif_set)
        # b, e, fitness, coverage, score
        print(candidate)
        b, e = int(candidate[0]), int(candidate[1])

        induced_paths = lcm.induced_paths(b, e)

        k = len(motif_set)

        series_flip = series.transpose().reshape(-1)
        motif_set_value = []
        for i in np.arange(k):
            current_sub_sequence = series_flip[motif_set[i][0]: motif_set[i][1]]
            motif_set_value.append(current_sub_sequence)

        motif_set_values_in_all_dimension = [motif_set_value]

        return series, motif_set_values_in_all_dimension, b, e, induced_paths

if __name__ == '__main__':
    motif_discovery = MotifDiscoveryUnivarite()
    series, motif_set_values_in_all_dimension, b, e, induced_paths = motif_discovery.datapreprocessing()
    representative_size = e - b
    n_of_motifs = len(motif_set_values_in_all_dimension[0])
    assoc_tabs, assoc_timeaxis_tabs = motif_discovery.mapping_preparing(series, induced_paths, n_of_motifs,
                                                                        representative_size, b)

    tsbubble_motif = TsBubble(representative_size)
    shifts_optimal = tsbubble_motif.find_the_optimal_shifts(assoc_timeaxis_tabs)
    for dim in np.arange(series.shape[1]):
        tsbubble_motif.plot_bubble_of_one_dimension(motif_set_values_in_all_dimension[dim], representative_size,
                                                    assoc_tabs[dim], assoc_timeaxis_tabs, n_of_motifs,
                                                    shifts_optimal)