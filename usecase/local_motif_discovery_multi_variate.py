#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from tsbubble.ts_bubble import  TsBubble
from motif_discovery import MotifDiscovery
# import altair as alt
#
# alt.data_transformers.enable('default', max_rows=None)
# alt.themes.enable('dark')

# get_ipython().run_line_magic('matplotlib', 'inline')

# plt.rcParams['figure.dpi'] = 200  # for both non-vector-graphics and rasterized vector-graphics figures
class MotifDiscoveryMultivarite(MotifDiscovery):

    def datapreprocessing(self, rho):
        PATH_DATASET = '../datasets/loco_motif_discovery/physical_therapy_dataset_Yurtman_Barshan'

        import locomotif.locomotif as locomotif
        import tsbubble.idle_segments as iseg

        subjects = range(5, 6)
        exercises = range(1, 9)
        template = False

        nb_motifs = 6
        l_min = 51
        l_max = 262
        overlap = 0.25
        restrict_starting_points = True
        subject = 5
        exercise = 2
        unit = 4 if exercise == 2 else 2
        # selected_column = 'acc_z' if exercise==2 else 'acc_x'
        # Load data:
        file_name = 'template_session' if template else 'test'
        FILE_NAME = f"s{subject}/e{exercise}/u{unit}/{file_name}.txt"
        data = pd.read_csv(f"{PATH_DATASET}/{FILE_NAME}", delimiter=';')
        data.rename(columns={data.columns[0]: 'timestamp'}, inplace=True)
        # series = data[selected_column]
        series = data[['acc_x', 'acc_y', 'acc_z']].values
        # series = data[['acc_z']].values
        # ds_name = FILE_NAME + ' ' + selected_column
        ds_name = f"subject {subject}, exercise {exercise}, unit {unit}"
        print(f'\n##### \n{ds_name}:\n')

        idle_mask = iseg.get_idle_mask(series, l_max, 0.005)
        start_mask = iseg.get_start_mask(series, idle_mask, 0.33)
        series = (series - np.mean(series, axis=0)) / np.std(series, axis=0)
        warping = True
        gamma = 1
        sm = locomotif.similarity_matrix_ndim(series, series, gamma, only_triu=False)
        tau = locomotif.estimate_tau_from_am(sm, rho)

        delta_a = -2 * tau
        delta_m = 0.5
        step_sizes = np.array([(1, 1), (2, 1), (1, 2)]) if warping else np.array([(1, 1)])
        lcm = locomotif.LoCoMotif(series=series, gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m,
                                  l_min=l_min,
                                  l_max=l_max, step_sizes=step_sizes)
        lcm._sm = sm
        lcm.align()
        lcm.kbest_paths(vwidth=l_min // 2)
        motif_sets = []
        for (candidate, motif_set), _ in lcm.kbest_motif_sets(nb=1, allowed_overlap=overlap,
                                                              start_mask=start_mask if restrict_starting_points else None,
                                                              end_mask=start_mask if restrict_starting_points else None):
            motif_sets.append(motif_set)
        # b, e, fitness, coverage, score
        print(candidate)
        b, e = int(candidate[0]), int(candidate[1])
        induced_paths = lcm.induced_paths(b, e)
        k = len(motif_set)
        series_flip = series.transpose().reshape(-1)
        motif_set_values_in_all_dimension = [[] for _ in np.arange(series.shape[1])]
        for dim in np.arange(series.shape[1]):
            motif_set_value = []
            for i in np.arange(k):
                current_sub_sequence = series_flip[
                                       motif_set[i][0] + dim * series.shape[0]: motif_set[i][1] + dim * series.shape[0]]
                motif_set_value.append(current_sub_sequence)
            motif_set_values_in_all_dimension[dim] = motif_set_value

        return series, motif_set_values_in_all_dimension, b, e, induced_paths

def find_lim_across_figure(rho_list):

    lim_for_time_series = [sys.float_info.max, sys.float_info.min]
    for rho in rho_list:
        motif_discovery = MotifDiscoveryMultivarite()
        series, motif_set_values_in_all_dimension, b, e, induced_paths = motif_discovery.datapreprocessing(rho)
        representative_size = e - b
        n_of_motifs = len(motif_set_values_in_all_dimension[0])
        assoc_tabs, assoc_timeaxis_tabs = motif_discovery.mapping_preparing(series, induced_paths, n_of_motifs,
                                                                            representative_size, b)

        tsbubble_motif = TsBubble()
        # shifts_optimal = tsbubble_motif.find_the_optimal_shifts(assoc_timeaxis_tabs)

        # pre process


        # lim_for_vertical_deviation = [sys.float_info.max, sys.float_info.min]

        # dtw_h_to_optimal_d, percent_h_o = \
        #     tsbubble_motif.get_dtw_horizontal_deviation_with_shifts(motif_set_values_in_all_dimension[0][0],
        #                                                             assoc_timeaxis_tabs, shifts_optimal)

        for dim in np.arange(series.shape[1]):

            for ts in motif_set_values_in_all_dimension[dim]:
                lim_for_time_series[1] = max(lim_for_time_series[1], max(ts))
                lim_for_time_series[0] = min(lim_for_time_series[0], min(ts))

            dtw_vertical_deviation, _ = tsbubble_motif.get_vertical_deviation_and_percent(
                motif_set_values_in_all_dimension[dim][0],
                assoc_tabs[dim])
            for i, timepoint in enumerate(motif_set_values_in_all_dimension[dim][0]):
                lim_for_time_series[1] = max(lim_for_time_series[1], timepoint + dtw_vertical_deviation[i])
                lim_for_time_series[0] = min(lim_for_time_series[0], timepoint - dtw_vertical_deviation[i])

            # lim_for_vertical_deviation[1] = max(lim_for_vertical_deviation[1], max(dtw_vertical_deviation))
            # lim_for_vertical_deviation[0] = min(lim_for_vertical_deviation[0], min(dtw_vertical_deviation))

    return lim_for_time_series

if __name__ == '__main__':
    rho_list = [0.4, 0.6, 0.7, 0.9]
    lim_for_time_series = find_lim_across_figure(rho_list)
    for rho in rho_list:
        save_fig_file_name = '/Users/ary/PycharmProjects/TsBubble/results/new_multivariate/' + str(rho) + ".png"
        motif_discovery = MotifDiscoveryMultivarite()
        series, motif_set_values_in_all_dimension, b, e, induced_paths = motif_discovery.datapreprocessing(rho)
        representative_size = e - b
        n_of_motifs = len(motif_set_values_in_all_dimension[0])
        assoc_tabs, assoc_timeaxis_tabs = motif_discovery.mapping_preparing(series, induced_paths, n_of_motifs,
                                                                        representative_size, b)

        tsbubble_motif = TsBubble()
        shifts_optimal = tsbubble_motif.find_the_optimal_shifts(assoc_timeaxis_tabs)

    #pre process

    # lim_for_time_series = [sys.float_info.max, sys.float_info.min]
    # lim_for_vertical_deviation = [sys.float_info.max, sys.float_info.min]

    # dtw_h_to_optimal_d, percent_h_o = \
    #     tsbubble_motif.get_dtw_horizontal_deviation_with_shifts(motif_set_values_in_all_dimension[0][0],
    #                                                             assoc_timeaxis_tabs, shifts_optimal)

    # for dim in np.arange(series.shape[1]):
    #
    #     for ts in motif_set_values_in_all_dimension[dim]:
    #         lim_for_time_series[1] = max(lim_for_time_series[1], max(ts))
    #         lim_for_time_series[0] = min(lim_for_time_series[0], min(ts))
    #
    #     dtw_vertical_deviation, _ = tsbubble_motif.get_vertical_deviation_and_percent(motif_set_values_in_all_dimension[dim][0],
    #                                                                                 assoc_tabs[dim])
    #     for i, timepoint in enumerate(motif_set_values_in_all_dimension[dim][0]):
    #         lim_for_time_series[1] = max(lim_for_time_series[1], timepoint+dtw_vertical_deviation[i])
    #         lim_for_time_series[0] = min(lim_for_time_series[0], timepoint-dtw_vertical_deviation[i])
    #
    #     lim_for_vertical_deviation[1] = max(lim_for_vertical_deviation[1], max(dtw_vertical_deviation))
    #     lim_for_vertical_deviation[0] = min(lim_for_vertical_deviation[0], min(dtw_vertical_deviation))

    #
    # for dim in np.arange(series.shape[1]):
    #     tsbubble_motif.plot_bubble_of_one_dimension(motif_set_values_in_all_dimension[dim], representative_size,
    #                                                 assoc_tabs[dim], assoc_timeaxis_tabs, n_of_motifs,
    #                                                 shifts_optimal, lim_for_time_series, lim_for_vertical_deviation)

        tsbubble_motif.plot_bubble_of_multi_dimension(motif_set_values_in_all_dimension, representative_size, assoc_tabs,
                                                  assoc_timeaxis_tabs, n_of_motifs, shifts_optimal, lim_for_time_series,
                                                  lim_for_vertical_deviation = None, save_fig_file_name=save_fig_file_name)




