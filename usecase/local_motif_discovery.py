import os

import numpy as np

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
l_max = int(  1 * fs)

# This parameter determines how much the motifs may overlap (intra and inter motif set)
overlap = 0
import locomotif.locomotif as locomotif

if series.ndim == 1:
    series = np.expand_dims(series, axis=1)

warping = True
gamma = 1
sm  = locomotif.similarity_matrix_ndim(series, series, gamma, only_triu=False)
tau = locomotif.estimate_tau_from_am(sm, rho)

delta_a = -2*tau
delta_m = 0.5
step_sizes = np.array([(1, 1), (2, 1), (1, 2)]) if warping else np.array([(1, 1)])

lcm = locomotif.LoCoMotif(series=series, gamma=gamma, tau=tau, delta_a=delta_a, delta_m=delta_m, l_min=l_min, l_max=l_max, step_sizes=step_sizes)
lcm._sm = sm
lcm.align()
lcm.kbest_paths(vwidth=l_min // 2)
motif_sets = []
for (candidate, motif_set), _ in lcm.kbest_motif_sets(nb=1, allowed_overlap=overlap, start_mask=None, end_mask=None):
    motif_sets.append(motif_set)

motif_set_length = len(motif_set)
# b, e, fitness, coverage, score
print(candidate)
b, e = int(candidate[0]), int(candidate[1])
length_of_candidate = e - b
import locomotif.visualize as visualize

fig, axs, _ = visualize.plot_sm(series, series, lcm._sm, matshow_kwargs={'alpha': 0.33})
visualize.plot_local_warping_paths(axs, lcm.get_paths(), lw=1)

induced_paths = lcm.induced_paths(b, e)
visualize.plot_local_warping_paths(axs, induced_paths, lw=3)

axs[3].axvline(b, lw=1, c='k', ls='--')
axs[3].axvline(e, lw=1, c='k', ls='--')

# z_norm = lambda orgin : ( orgin - np.mean(orgin) ) / np.std(orgin)

import matplotlib.pyplot as plt
k = len(motif_set)
fig, ax = plt.subplots(2, k + 1, figsize=(2 * (0.5 + k), 2 * 1.5), width_ratios=[0.5] + + k * [1],
                       height_ratios=[0.5, 1])
ax[0, 0].set_axis_off()
candidate_subsq = series[b:e, :]
ax[1, 0].plot(-candidate_subsq, range(len(candidate_subsq), 0, -1))
ax[1, 0].set_ylim([-0.5, e - b + 0.5])
ax[1, 0].set_axis_off()

assoc_tab_new_paths = [[[] for _ in range(motif_set_length)] for _ in range(length_of_candidate)]
assoc_timeaxis_tab_new_paths = [[[] for _ in range(motif_set_length)] for _ in range(length_of_candidate)]


series_flip = series.transpose().reshape(-1)
motif_set_value = []

selected_sub_sequence_id = 0
#z_normaliza the data
for i in np.arange(k):
    current_sub_sequence = series_flip[motif_set[i][0]: motif_set[i][1]]
    z_norm_current_sub_sequence = \
        (current_sub_sequence - np.mean(current_sub_sequence)) / np.std(current_sub_sequence)
    motif_set_value.append(current_sub_sequence)
    # motif_set_value.append(z_norm_current_sub_sequence)
motif_set_value

max_length = e - b
for i, path in enumerate(induced_paths):
    (bm, em) = path[0][0],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   path[-1][0]
    if em - bm > max_length :
        max_length = em - bm
    ax[0, i + 1].plot(series[bm:em, :])
    ax[0, i + 1].set_xlim([-0.5, em - bm + 0.5])
    ax[0, i + 1].set_axis_off()

    ax[1, i + 1].invert_yaxis()
    ax[1, i + 1].plot(path[:, 0] - bm, path[:, 1] - b, c='r', ls='-', marker='.', markersize=1)
    ax[1, i + 1].set_ylim([e - b, 0])
    ax[1, i + 1].set_xlim([0, em - bm])

    for mapping in path:
        assoc_timeaxis_tab_new_paths[mapping[1] - b][i].append(mapping[0] - bm)
        assoc_tab_new_paths[mapping[1] - b][i].append(series_flip[mapping[0]])

S_i  = [0] * len(induced_paths)
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
        H_i_t = assoc_timeaxis_tab_new_paths[t][i]
        current_different = np.sum(t - H_i_t)
        current_align_cnt = len(H_i_t)
        shift_top += 1 / H_t[t] * current_different
        shift_denominator += 1 / H_t[t] * current_align_cnt
    S_i[i] = shift_top / shift_denominator



##plot the overlayying warping paths:
fig_overlayed, ax_overlayed = plt.subplots(2, 4, figsize=(12, 4))
ax_overlayed[0, 0].set_axis_off()

ax_overlayed[1, 0].plot(-candidate_subsq, range(len(candidate_subsq), 0, -1))
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

ax_overlayed[0, 3].set_xlim([(e - b) / 2 + (-max_length / 2 - 0.5) , (e - b) / 2 + (max_length /  2 + 0.5)])
ax_overlayed[0, 3].set_axis_off()

ax_overlayed[1, 3].invert_yaxis()
ax_overlayed[1, 3].set_ylim([e - b, 0])
ax_overlayed[1, 3].set_xlim([(e - b) / 2 + (-max_length / 2 - 0.5) , (e - b) / 2 + (max_length /  2 + 0.5)])

#
# ax_overlayed[0, 4].set_xlim([(e - b) / 2 + (-max_length / 2 - 0.5) , (e - b) / 2 + (max_length /  2 + 0.5)])
# ax_overlayed[0, 4].set_axis_off()
#
# ax_overlayed[1, 4].invert_yaxis()
# ax_overlayed[1, 4].set_ylim([e - b, 0])
# ax_overlayed[1, 4].set_xlim([(e - b) / 2 + (-max_length / 2 - 0.5) , (e - b) / 2 + (max_length /  2 + 0.5)])

alpha = 0.6

shifts_right = []
shifts_middle = []

for i, path in enumerate(induced_paths):
    (bm, em) = path[0][0], path[-1][0]
    ax_overlayed[0, 1].plot(series[bm:em, :],  alpha = alpha)
    ax_overlayed[1, 1].plot(path[:, 0] - bm, path[:, 1] - b, ls='-', marker='.', markersize=1, alpha = alpha)

    shift_to_right = (e - b) -  em
    shifts_right.append(bm + shift_to_right)
    ax_overlayed[0, 2].plot(np.arange(bm + shift_to_right, em + shift_to_right), series[bm:em, :], alpha = alpha)
    ax_overlayed[1, 2].plot(path[:, 0] + shift_to_right, path[:, 1] - b, ls ='-', marker ='.', markersize = 1, alpha = alpha)

    shift_to_middle = (e-b) / 2 - (bm + em) / 2
    shifts_middle.append(bm + shift_to_middle)
    ax_overlayed[0, 3].plot(np.arange(bm + shift_to_middle, em + shift_to_middle), series[bm:em, :], alpha = alpha)
    ax_overlayed[1, 3].plot(path[:, 0] + shift_to_middle, path[:, 1] - b, ls='-', marker='.', markersize=1, alpha=alpha)

# from examples.deprecated.explain_kmeans_dtw import ExplainKmeans
from kmeans_dba import ExplainKmeans

explain = ExplainKmeans(series = motif_set_value, span = (0, length_of_candidate), max_iter=None, k=1, dist_matrix=None, V = None, cluster_and_idx= None, align_info_provided = True)
dtw_vertical_deviation, percent_v = explain.get_vertical_deviation_and_percent(motif_set_value[0], assoc_tab_new_paths)
dtw_horizontal_deviation, percent_h_l = \
    explain.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths, np.zeros(len(motif_set_value)))
dtw_h_to_right_d, percent_h_r = \
    explain.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths, shifts_right)
dtw_h_to_middle_d, percent_h_m = \
    explain.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths, shifts_middle)
dtw_h_to_optimal_d, percent_h_o = \
    explain.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths, S_i)


ax1, ax2, ax3, ax4, ax5, ax6 = ExplainKmeans.init_figure_for_different_deviations()
explain.plotAllSeries(ax1,  motif_set_value, motif_set_length)

ax6.plot(explain.x, dtw_vertical_deviation, color='black', label='VWD',linewidth=0.3)
ax6_twin = ax6.twinx()
ax6_twin.plot(explain.x, dtw_horizontal_deviation, color='green', label='HWD_to_left', linewidth=0.3)
ax6_twin.plot(explain.x, dtw_h_to_right_d, color ='blue', label ='HWD_to_right', linewidth = 0.3)
ax6_twin.plot(explain.x, dtw_h_to_middle_d, color ='red', label ='HWD_to_middle', linewidth = 0.3)
ax6_twin.plot(explain.x, dtw_h_to_optimal_d, color = 'purple', label = 'HWD_optimal', linewidth = 2)

explain.plot_eclipse_and_percent_around_dba(ax2, motif_set_value[0], dtw_horizontal_deviation, dtw_vertical_deviation, percent_v, percent_h_l, False)
explain.plot_eclipse_and_percent_around_dba(ax3, motif_set_value[0], dtw_h_to_right_d, dtw_vertical_deviation, percent_v, percent_h_r, False)
explain.plot_eclipse_and_percent_around_dba(ax4, motif_set_value[0], dtw_h_to_middle_d, dtw_vertical_deviation, percent_v, percent_h_m, False)
explain.plot_eclipse_and_percent_around_dba(ax5, motif_set_value[0], dtw_h_to_optimal_d, dtw_vertical_deviation, percent_v, percent_h_o, False)
plt.legend(loc = 'best')
plt.show()

# while True:
#     centroid_id = int(input("centroid index to plot the point cloud:"))
#     y_values = sum(assoc_tab_new_paths[centroid_id], [])
#     x_values = sum(assoc_timeaxis_tab_new_paths[centroid_id], [])
#     plt.figure()
#     plt.scatter(x_values, y_values, color='grey')
#     plt.scatter(centroid_id, motif_set_value[0][centroid_id], color="red")
#     plt.show()