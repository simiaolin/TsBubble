#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pathlib import Path
import scipy as scp

from examples.deprecated.explain_kmeans_dtw import ExplainKmeans
from dtaidistance.dtw import warping_path
import dtaidistance.dtw_visualisation as dtwvis
import sys
import altair as alt

alt.data_transformers.enable('default', max_rows=None)
alt.themes.enable('dark')

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200  # for both non-vector-graphics and rasterized vector-graphics figures
PATH_DATASET = '../datasets/loco_motif_discovery/physical_therapy_dataset_Yurtman_Barshan'

import locomotif.locomotif as locomotif
from locomotif.locomotif import LoCoMotif, similarity_matrix_ndim, estimate_tau_from_am
from locomotif.visualize import plot_motif_sets
import tsbuble.idle_segments as iseg

subjects = range(5, 6)
exercises = range(1, 9)
template = False
rho = 0.7
nb_motifs = 6
l_min = 51
l_max = 262
overlap = 0.25
restrict_starting_points = True


def plot_all_the_alignments(series, candidate_subsq, e, b, max_length, induced_paths, assoc_timeaxis_tab_new_paths):
    alpha = 0.6
    shifts_right, shifts_middle = find_right_and_middle_shifts(induced_paths, e, b)
    shifts_optimal = find_the_optimal_shifts(induced_paths, assoc_timeaxis_tab_new_paths)

    for dim in np.arange(series.shape[1]): #dimension
        fig_overlayed, ax_overlayed = init_overlaying_alignment_figure(e, b, max_length, shifts_optimal, induced_paths)
        ax_overlayed[1, 0].plot(-candidate_subsq[:, dim], range(len(candidate_subsq), 0, -1))

        for i, path in enumerate(induced_paths):
            (bm, em) = path[0][0], path[-1][0]
            ax_overlayed[0, 1].plot(series [bm:em, dim], alpha=alpha)
            ax_overlayed[1, 1].plot(path[:, 0] - bm, path[:, 1] - b, ls='-', marker='.', markersize=1, alpha=alpha)

            ax_overlayed[0, 2].plot(np.arange(shifts_right[i], em - bm + shifts_right[i]), series[bm:em, dim], alpha=alpha)
            ax_overlayed[1, 2].plot(path[:, 0] + shifts_right[i] - bm, path[:, 1] - b, ls='-', marker='.', markersize=1,
                                    alpha=alpha)

            ax_overlayed[0, 3].plot(np.arange(shifts_middle[i], em - bm + shifts_middle[i]), series[bm:em, dim], alpha=alpha)
            ax_overlayed[1, 3].plot(path[:, 0] + shifts_middle[i] - bm, path[:, 1] - b, ls='-', marker='.', markersize=1,
                                    alpha=alpha)

            ax_overlayed[0, 4].plot(np.arange(shifts_optimal[i], em - bm + shifts_optimal[i]), series[bm:em, dim], alpha=alpha)
            ax_overlayed[1, 4].plot(path[:, 0] + shifts_optimal[i] - bm, path[:, 1] - b, ls='-', marker='.', markersize=1,
                                    alpha=alpha)


    plt.show()
    return shifts_right, shifts_middle, shifts_optimal

def init_overlaying_alignment_figure( e, b, max_length, shifts_optimal, induced_paths):
    fig_overlayed, ax_overlayed = plt.subplots(2, 5, figsize=(15, 4))
    ax_overlayed[0, 0].set_axis_off()

    ax_overlayed[1, 0].set_ylim([-0.5, e - b + 0.5])
    ax_overlayed[1, 0].set_axis_off()

    ax_overlayed[0, 1].set_xlim([-0.5, max_length + 0.5])
    ax_overlayed[0, 1].set_axis_off()

    ax_overlayed[1, 1].invert_yaxis()
    ax_overlayed[1, 1].set_ylim([e - b, 0])
    ax_overlayed[1, 1].set_xlim([-0.5, max_length + 0.5])

    ax_overlayed[0, 2].set_xlim([e - b - max_length - 0.5, e - b + 0.5])
    ax_overlayed[0, 2].set_axis_off()

    ax_overlayed[1, 2].invert_yaxis()
    ax_overlayed[1, 2].set_ylim([e - b, 0])
    ax_overlayed[1, 2].set_xlim([e - b - max_length - 0.5, e - b + 0.5])

    ax_overlayed[0, 3].set_xlim([(e - b) / 2 + (-max_length / 2 - 0.5), (e - b) / 2 + (max_length / 2 + 0.5)])
    ax_overlayed[0, 3].set_axis_off()

    ax_overlayed[1, 3].invert_yaxis()
    ax_overlayed[1, 3].set_ylim([e - b, 0])
    ax_overlayed[1, 3].set_xlim([(e - b) / 2 + (-max_length / 2 - 0.5), (e - b) / 2 + (max_length / 2 + 0.5)])


    left_most = sys.float_info.max
    right_most = sys.float_info.min

    for i, path in enumerate(induced_paths):
        (bm, em) = path[0][0], path[-1][0]
        if shifts_optimal[i] < left_most:
            left_most = shifts_optimal[i]
        if em - bm + shifts_optimal[i] > right_most:
            right_most = em - bm  +  shifts_optimal[i]


    ax_overlayed[0, 4].set_xlim([left_most - 0.5, right_most +  0.5])
    ax_overlayed[0, 4].set_axis_off()
    ax_overlayed[1, 4].invert_yaxis()
    ax_overlayed[1, 4].set_ylim([e - b, 0])
    ax_overlayed[1, 4].sharex(ax_overlayed[0, 4])

    return fig_overlayed, ax_overlayed

def find_right_and_middle_shifts(induced_paths, e, b):
    shifts_right = []
    shifts_middle = []
    for i, path in enumerate(induced_paths):
        (bm, em) = path[0][0], path[-1][0]
        shift_to_right = (e - b) - em
        shifts_right.append(bm + shift_to_right)

        shift_to_middle = (e - b) / 2 - (bm + em) / 2
        shifts_middle.append(bm + shift_to_middle)
    return shifts_right, shifts_middle

def find_the_optimal_shifts(induced_paths, assoc_timeaxis_tab_new_paths):
    S_i = [0] * len(induced_paths)
    H_t = [0] * len(assoc_timeaxis_tab_new_paths)
    for i in np.arange(len(assoc_timeaxis_tab_new_paths)):
        alignment_cnt = 0
        for t in assoc_timeaxis_tab_new_paths[i]:
            alignment_cnt += len(t)
        H_t[i] = alignment_cnt

    for i in np.arange(len(induced_paths)):
        shift_top = 0
        shift_denominator = 0
        for t in np.arange(len(H_t)):
            if H_t[t] != 0:

                H_i_t = assoc_timeaxis_tab_new_paths[t][i]
                current_different = np.sum(t - H_i_t)
                current_align_cnt = len(H_i_t)

                shift_top += 1 / H_t[t] * current_different
                shift_denominator += 1 / H_t[t] * current_align_cnt


        S_i[i] = shift_top / shift_denominator
    return S_i

def plot_bubble_of_one_dimension(motif_set_value, length_of_candidate,
                                 assoc_tab_new_paths, assoc_timeaxis_tab_new_paths, motif_set_length, shifts_optimal):

    explain = ExplainKmeans(series=motif_set_value, span=(0, length_of_candidate - 1), max_iter=None, k=1,
                            dist_matrix=None, V=None, cluster_and_idx=None)
    dtw_vertical_deviation, percent_v = explain.get_dtw_vertical_deviation_and_percent(motif_set_value[0],
                                                                                       assoc_tab_new_paths)
    dtw_h_to_optimal_d, percent_h_o = \
        explain.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths, shifts_optimal)
    fig, axs = plt.subplots(3, 1, dpi=200, sharex=True, figsize=(10, 10))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    ax1.set_title("All Instances", color="blue")
    ax2.set_title("Bubbles (warping to optimal)", color='blue', size=5)
    ax3.set_title("All types of deviations", color="red", size=5)
    explain.plotAllSeries_with_optimal_shifts(ax1, motif_set_value, motif_set_length, shifts_optimal)

    explain.plot_eclipse_and_percent_around_dba(ax2, motif_set_value[0], dtw_h_to_optimal_d, dtw_vertical_deviation,
                                                percent_v, percent_h_o, False)

    ax3.plot(explain.x, dtw_vertical_deviation, color='black', label='VWD', linewidth=0.3)
    ax3_twin = ax3.twinx()
    ax3_twin.spines['right'].set_color('purple')
    ax3_twin.spines['right'].set_linewidth(4)
    ax3_twin.plot(explain.x, dtw_h_to_optimal_d, color='purple', label='HWD_optimal', linewidth=2)
    plt.legend(loc='best')
    plt.show()

def main():
    for i, subject in enumerate(subjects):
        for j, exercise in enumerate(exercises):
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

            # Find idle time intervals and restrict the starting points of the motifs:
            idle_segments = iseg.get_idle_segments(series, l_max, threshold=0.005)

            idle_mask = iseg.get_idle_mask(series, l_max, 0.005)
            start_mask = iseg.get_start_mask(series, idle_mask, 0.33)

            fig, ax = plt.subplots(figsize=(14, 2))
            ax.plot(series)
            for (s, e) in idle_segments:
                ax.axvspan(s, e, color='red', alpha=0.5)
            for (s, e) in iseg.get_segments(start_mask):
                ax.axvspan(s, e, color='green', alpha=0.5)
            plt.title(ds_name)
            plt.show()

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
            motif_set_length = len(motif_set)
            # b, e, fitness, coverage, score
            print(candidate)
            b, e = int(candidate[0]), int(candidate[1])
            length_of_candidate = e - b + 1
            import locomotif.visualize as visualize
            fig, axs, _ = visualize.plot_sm(series, series, lcm._sm, matshow_kwargs={'alpha': 0.33})
            visualize.plot_local_warping_paths(axs, lcm.get_paths(), lw=1)
            induced_paths = lcm.induced_paths(b, e)
            visualize.plot_local_warping_paths(axs, induced_paths, lw=3)
            axs[3].axvline(b, lw=1, c='k', ls='--')
            axs[3].axvline(e, lw=1, c='k', ls='--')
            # z_norm = lambda orgin : ( orgin - np.mean(orgin) ) / np.std(orgin)

            k = len(motif_set)
            fig, ax = plt.subplots(2, k + 1, figsize=(2 * (0.5 + k), 2 * 1.5), width_ratios=[0.5] + + k * [1],
                                   height_ratios=[0.5, 1])
            ax[0, 0].set_axis_off()
            candidate_subsq = series[b:e, :]
            ax[1, 0].plot(-candidate_subsq, range(len(candidate_subsq), 0, -1))
            ax[1, 0].set_ylim([-0.5, e - b + 0.5])
            ax[1, 0].set_axis_off()

            assoc_tab_new_paths = [[[[] for _ in range(motif_set_length)] for _ in range(length_of_candidate)] for _ in range(series.shape[1])]
            assoc_timeaxis_tab_new_paths = [[[] for _ in range(motif_set_length)] for _ in range(length_of_candidate)]

            series_flip = series.transpose().reshape(-1)
            motif_set_values_in_all_dimension = [[] for i in np.arange(series.shape[1])]
            for dim in np.arange(series.shape[1]):
                motif_set_value = []
                for i in np.arange(k):
                    current_sub_sequence = series_flip[motif_set[i][0] + dim * series.shape[0]: motif_set[i][1] + dim * series.shape[0]]
                    motif_set_value.append(current_sub_sequence)
                motif_set_values_in_all_dimension[dim] = motif_set_value

            max_length = e - b
            for i, path in enumerate(induced_paths):
                (bm, em) = path[0][0], path[-1][0]
                if em - bm > max_length:
                    max_length = em - bm
                ax[0, i + 1].plot(series[bm:em, :])  #all the dimensions will be plotted.
                ax[0, i + 1].set_xlim([-0.5, em - bm + 0.5])
                ax[0, i + 1].set_axis_off()
                ax[1, i + 1].invert_yaxis()
                ax[1, i + 1].plot(path[:, 0] - bm, path[:, 1] - b, c='r', ls='-', marker='.', markersize=1)
                ax[1, i + 1].set_ylim([e - b, 0])
                ax[1, i + 1].set_xlim([0, em - bm])
                for mapping in path:
                    assoc_timeaxis_tab_new_paths[mapping[1] - b][i].append(mapping[0] - bm)
                    for dim in np.arange(series.shape[1]):
                        assoc_tab_new_paths[dim][mapping[1] - b][i].append(series_flip[mapping[0] + dim * series.shape[0]])
            plt.show()

            shifts_right, shifts_middle, shifts_optimal = \
                plot_all_the_alignments(series, candidate_subsq, e, b, max_length, induced_paths,
                                        assoc_timeaxis_tab_new_paths)
            for dim in np.arange(series.shape[1]):
                plot_bubble_of_one_dimension(motif_set_values_in_all_dimension[dim], length_of_candidate,
                                         assoc_tab_new_paths[dim], assoc_timeaxis_tab_new_paths, motif_set_length,
                                          shifts_optimal)


if __name__ == '__main__':
    main()


# deprecated
def get_all_alignment_dtw():
    ...
    # assoc_tab = [[[] for _ in range(17)] for _ in range(100)]
    # assoc_timeaxis_tab = [[[] for _ in range(17)] for _ in range(100)]
    # current_idx = 0
    # original_idx = []
    # m = warping_path(motif_set_value[0], motif_set_value[7])
    # dtwvis.plot_warping(motif_set_value[0], motif_set_value[7], m, filename="/Users/ary/Desktop/warping_paths/" + str(7) + ".png")
    #
    # for idx, seq in enumerate(motif_set_value):
    #     m = warping_path(motif_set_value[0], seq)
    #     # dtwvis.plot_warping(motif_set_value[0], seq, m, filename="/Users/ary/Desktop/warping_paths/" + str(idx) + ".png")
    #
    #     for i, j in m:
    #         assoc_tab[i][current_idx].append(seq[j])
    #         assoc_timeaxis_tab[i][current_idx].append(j)
    #     current_idx += 1
    #     original_idx.append(idx)


# deprecated
def test_the_optimicity_of_shifts(S_i):
    ...
    # import sys
    # d1 = np.sum(np.square(dtw_horizontal_deviation))
    # d2 = np.sum(np.square(dtw_h_to_right_d))
    # d3 = np.sum(np.square(dtw_h_to_middle_d))
    # d4 = np.sum(np.square(dtw_h_to_optimal_d))
    #
    # #or checking whether the one we find is really the optimal one by adding some noises
    # import copy
    # S_i_with_noise = []
    # for t in np.arange(len(S_i)):
    #     noises = np.random.uniform(low=-5, high=5, size=(100,))
    #     for n in noises:
    #         S_new = copy.deepcopy(S_i)
    #         S_new[t] += n
    #         S_i_with_noise.append(S_new)
    # differences = []
    # for s_i_with_noise in S_i_with_noise:
    #     dtw_warping = explain.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths, s_i_with_noise)
    #     sum_of_square_deviation = np.sum(np.square(dtw_warping))
    #     differences.append(sum_of_square_deviation - d4)
    # biggest_d = sys.float_info.min
    # smallest_d = sys.float_info.max
    #
    # for d in differences:
    #     assert d > 0
    #     if d > biggest_d:
    #         biggest_d = d
    #
    #     if d < smallest_d:
    #         smallest_d = d

def plot_bubble_of_one_dimension_old(motif_set_value, length_of_candidate,
                                 assoc_tab_new_paths, assoc_timeaxis_tab_new_paths, motif_set_length,
                                 shifts_right, shifts_middle, S_i):
    explain = ExplainKmeans(series=motif_set_value, span=(0, length_of_candidate), max_iter=None, k=1,
                            dist_matrix=None, V=None, cluster_and_idx=None)
    dtw_vertical_deviation, percent_v = explain.get_dtw_vertical_deviation_and_percent(motif_set_value[0],
                                                                                       assoc_tab_new_paths)
    dtw_horizontal_deviation, percent_h_l = \
        explain.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths,
                                                         np.zeros(len(motif_set_value)))
    dtw_h_to_right_d, percent_h_r = \
        explain.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths,
                                                         shifts_right)
    dtw_h_to_middle_d, percent_h_m = \
        explain.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths,
                                                         shifts_middle)
    dtw_h_to_optimal_d, percent_h_o = \
        explain.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths, S_i)

    ax1, ax2, ax3, ax4, ax5, ax6 = ExplainKmeans.init_figure_for_different_deviations()
    explain.plotAllSeries(ax1, motif_set_value, motif_set_length)
    # explain.plot_adjusted_series(ax2, 0, assoc_timeaxis_tab_new_paths, original_idx, motif_set_value)
    # explain.plot_statistics_curves(ax4, dtw_vertical_deviation, None, dtw_horizontal_deviation)
    ax6.plot(explain.x, dtw_vertical_deviation, color='black', label='VWD', linewidth=0.3)
    ax6_twin = ax6.twinx()
    ax6_twin.plot(explain.x, dtw_horizontal_deviation, color='green', label='HWD_to_left', linewidth=0.3)
    ax6_twin.plot(explain.x, dtw_h_to_right_d, color='blue', label='HWD_to_right', linewidth=0.3)
    ax6_twin.plot(explain.x, dtw_h_to_middle_d, color='red', label='HWD_to_middle', linewidth=0.3)
    ax6_twin.plot(explain.x, dtw_h_to_optimal_d, color='purple', label='HWD_optimal', linewidth=2)

    explain.plot_eclipse_and_percent_around_dba(ax2, motif_set_value[0], dtw_horizontal_deviation,
                                                dtw_vertical_deviation, percent_v, percent_h_l, False)
    explain.plot_eclipse_and_percent_around_dba(ax3, motif_set_value[0], dtw_h_to_right_d, dtw_vertical_deviation,
                                                percent_v, percent_h_r, False)
    explain.plot_eclipse_and_percent_around_dba(ax4, motif_set_value[0], dtw_h_to_middle_d, dtw_vertical_deviation,
                                                percent_v, percent_h_m, False)
    explain.plot_eclipse_and_percent_around_dba(ax5, motif_set_value[0], dtw_h_to_optimal_d, dtw_vertical_deviation,
                                                percent_v, percent_h_o, False)
    plt.legend(loc='best')
    plt.show()

    # while True:
    #     centroid_id = int(input("centroid index to plot the point cloud:"))
    #     y_values = sum(assoc_tab_new_paths[centroid_id], [])
    #     x_values = sum(assoc_timeaxis_tab_new_paths[centroid_id], [])
    #     plt.figure()
    #     plt.scatter(x_values, y_values, color='grey')
    #     plt.scatter(centroid_id, motif_set_value[0][centroid_id], color="red")
    #     plt.show()

