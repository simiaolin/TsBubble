import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.text  import Text
import matplotlib.cm as cm
import matplotlib
font = {
                # 'weight': 'bold',
                'size': 22}
figure_conf = {
    'weight':'bold',
    'titlesize':30
}
matplotlib.rc('font', **font)
matplotlib.rc('figure', titlesize=30)
# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
import bisect

class AlignmentInfo:
    def __init__(self, series_mean, original_idx, assoc_tab, assoc_timeaxis_tab, cur_series):
        self.series_mean = series_mean
        self.original_idx = original_idx
        self.assoc_tab = assoc_tab
        self.assoc_timeaxis_tab = assoc_timeaxis_tab
        self.cur_series = cur_series

class TsBubble():

    def find_the_optimal_shifts(self, assoc_timeaxis_tab_new_paths):
        number_of_instances = len(assoc_timeaxis_tab_new_paths[0])
        S_i = [0] * number_of_instances
        H_t = [0] * len(assoc_timeaxis_tab_new_paths)
        for i in np.arange(len(assoc_timeaxis_tab_new_paths)):
            alignment_cnt = 0
            for t in assoc_timeaxis_tab_new_paths[i]:
                alignment_cnt += len(t)
            H_t[i] = alignment_cnt

        for i in np.arange(number_of_instances):
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

        S_i.insert(0, 0.0)
        return S_i

    def find_the_optimal_independent_shifts(self, assoc_timeaxis_tab_new_paths):
        number_of_instances = len(assoc_timeaxis_tab_new_paths[0])
        S_i = [0] * number_of_instances
        for i in np.arange(number_of_instances):

            sum_difference = 0
            cnt_difference = 0
            for t in np.arange(len(assoc_timeaxis_tab_new_paths)):
                current_mapping = assoc_timeaxis_tab_new_paths[t][i]
                sum_difference += np.sum(t - current_mapping)
                cnt_difference += len(current_mapping)

            S_i[i] = sum_difference / cnt_difference
        S_i.insert(0, 0.0)
        return S_i

    def get_vertical_deviation_and_percent(self, series_mean, assoc_tab):   ##should use the old mean or the new mean?
        dtw_vertical_deviation = np.empty(shape=series_mean.shape)
        dtw_v_percent = np.empty(shape=series_mean.shape)
        for i in range(len(series_mean)):
            all_values_aligned_to_current_idx = []
            for _, values in enumerate(assoc_tab[i]):
                all_values_aligned_to_current_idx = np.append(all_values_aligned_to_current_idx, values)
            current_div = np.sqrt(np.divide(
                np.sum(np.square(all_values_aligned_to_current_idx - series_mean[i])),
                len(all_values_aligned_to_current_idx)
                ))
            inside_range_cnt = 0
            for v in  all_values_aligned_to_current_idx:
                if series_mean[i] - current_div <= v <= series_mean[i] + current_div:
                    inside_range_cnt += 1
            dtw_vertical_deviation[i] = current_div
            dtw_v_percent[i] = inside_range_cnt / len(all_values_aligned_to_current_idx)
        return dtw_vertical_deviation, dtw_v_percent

    def get_dtw_horizontal_deviation_with_shifts(self, series_mean, assoc_timeaxis_tab, shifts):  #get the horizontal deviation with a shift
        dtw_horizontal_deviation = np.empty(shape=series_mean.shape)
        inside_one_deviation_percentage = np.empty(shape=series_mean.shape)
        for i in range(len(series_mean)):
            all_timeaxis_assigned_to_current_idx = []
            for j, idxes in enumerate(assoc_timeaxis_tab[i]):
                all_timeaxis_assigned_to_current_idx = np.append(all_timeaxis_assigned_to_current_idx, idxes + shifts[j+1])
            current_div = np.sqrt(
                np.divide(
                    np.sum(np.square(all_timeaxis_assigned_to_current_idx - i)),
                    len(all_timeaxis_assigned_to_current_idx)
                    )
                )
            inside_range_cnt = 0
            for assign_time_axis in all_timeaxis_assigned_to_current_idx:
                if i - current_div <= assign_time_axis <= i + current_div:
                    inside_range_cnt += 1
            dtw_horizontal_deviation[i] = current_div
            inside_one_deviation_percentage[i] = inside_range_cnt / len(all_timeaxis_assigned_to_current_idx)
        return dtw_horizontal_deviation, inside_one_deviation_percentage

    def get_color_iter(self, size):
        color_arr = cm.rainbow(np.linspace(0, 1, size))
        colors = iter(color_arr)
        return colors


    def plotAllSeries_with_optimal_shifts(self, plt,  motif_set_value, size, shifts_optimal):
        colors = self.get_color_iter(size)  ## the size of all the series that are used to calculate the mean

        for new_idx in range(size):
            if new_idx == 0: # the centroid of the local motif case
                ...
                # no longer plot the representative
                # x = np.arange(0, len(motif_set_value[new_idx]))
                # plt.plot(x, motif_set_value[new_idx], color = 'purple', linewidth = 2)
            else:
                plt.plot(shifts_optimal[new_idx] + np.arange(0, len(motif_set_value[new_idx])),  motif_set_value[new_idx], color=next(colors) ,linewidth = 0.3)
    def get_elipse_index_list(self, vwd, hwd, order, series_mean=None): #horizontal wraping deviation, vertical wraping deviation
        n = len(hwd)
        area = list(map(lambda i: hwd[i] * vwd[i],  range(n)))
        sort_index = np.argsort(area)   # small elipses rank first
        if not order:
            sort_index = np.flip(sort_index)  #bigger elipses rank first
        occupied_squares = []
        elipse_index_list = []
        for i in sort_index:
            self.try_insert_elipse(occupied_squares, new_index=i, value=series_mean[i], r_height=vwd[i], r_width=hwd[i], elipse_index_list=elipse_index_list)
        return elipse_index_list

    def try_insert_elipse(self, occupied_squares, new_index, value, r_height, r_width, elipse_index_list):
        cur_left_most = new_index - r_width
        cur_right_most = new_index + r_width

        cur_up_most = value + r_height
        cur_bottom_most = value - r_height

        # index = bisect.bisect_right(occupied_left_index_list, left)
        #
        # last_right = occupied_right_index_list[index - 1]
        # next_left = occupied_left_index_list[index]
        if len(occupied_squares) == 0:
            occupied_squares.append([cur_left_most, cur_right_most, cur_up_most, cur_bottom_most])
            elipse_index_list.append(new_index)
        else:

            flag = True
            for covered_square in occupied_squares:
                cur_corners = [(new_index - r_width, value - r_height),
                               (new_index + r_width, value - r_height),
                               (new_index - r_width, value + r_height),
                               (new_index + r_width, value + r_height)
                               ]
                for cur_corner in cur_corners:
                    if  covered_square[0] < cur_corner[0] < covered_square[1] and covered_square[3] < cur_corner[1] < covered_square[2]:
                        flag = False
                        break
            if flag == True:
                occupied_squares.append([cur_left_most, cur_right_most, cur_up_most, cur_bottom_most])
                elipse_index_list.append(new_index)

    def plot_eclipse_and_percent_around_dba(self, plt, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation, v_percent, h_percent, order,  percentageOn = False):  #plotting elipses without overlapping among them.
        plt.plot(series_mean, color='purple' ,linewidth = 2)
        color_type_num = 10
        color_arr = cm.rainbow(np.linspace(0, 1, color_type_num))
        elipse_index_list = self.get_elipse_index_list(vwd=dtw_vertical_deviation, hwd=dtw_horizontal_deviation,
                                                       order=order, series_mean=series_mean)
        ells = [Ellipse(xy=(i, series_mean[i]),
                        width=dtw_horizontal_deviation[i], height=dtw_vertical_deviation[i], color=color_arr[i%color_type_num])
                for i in elipse_index_list
                ]


        for e in ells:
            plt.add_artist(e)
        if percentageOn:
            textlist = [Text(i, series_mean[i],
                             text='h:' + "%.2f" % h_percent[i] + '\nv:' + "%.2f" % v_percent[i])
                        for i in elipse_index_list]
            for t in textlist:
                plt.add_artist(t)

    def plot_eclipse_around_dba_of_different_dimensions(self, plt, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation, order):  #plotting elipses without overlapping among them.
        detected_dim = len(series_mean)
        color_arr = cm.brg(np.linspace(0, 1, detected_dim))
        for dim in np.arange(detected_dim):
            plt.plot(series_mean[dim], color=color_arr[dim], zorder = 0)
        for dim in np.arange(detected_dim):
            elipse_index_list = self.get_elipse_index_list(vwd=dtw_vertical_deviation[dim],
                                                           hwd=dtw_horizontal_deviation, order=order, series_mean=series_mean[dim])
            ells = [Ellipse(xy=(i, series_mean[dim][i]),
                        width=dtw_horizontal_deviation[i], height=dtw_vertical_deviation[dim][i], color=color_arr[dim], alpha = 0.3, zorder=1)
                for i in elipse_index_list
                ]
            for e in ells:
                plt.add_artist(e)


    def plot_bubble_of_one_dimension(self, motif_set_value, length_of_candidate, assoc_tab_new_paths,
                                     assoc_timeaxis_tab_new_paths, motif_set_length, shifts_optimal, save_fig_name=None,
                                     lim_for_time_series=None):
        # explain = ExplainKmeans(series=motif_set_value, span=(0, length_of_candidate - 1), max_iter=None, k=1,
        #                         dist_matrix=None, V=None, cluster_and_idx=None, align_info_provided=True)
        dtw_vertical_deviation, percent_v = self.get_vertical_deviation_and_percent(motif_set_value[0],
                                                                                       assoc_tab_new_paths)
        dtw_h_to_optimal_d, percent_h_o = \
            self.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths, shifts_optimal)

        # variance_of_optimal_shifts = np.sum(np.square(dtw_h_to_optimal_d))

        # dtw_h_no_shifts, percent_h_o = \
        #     self.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths,
        #                                                   [np.float64(0)] * len(shifts_optimal))
        # variance_of_no_shifts = np.sum(np.square(dtw_h_no_shifts))
        # print("the variance of optimal shifts: " + str(variance_of_optimal_shifts))

        # print("the variance without shifts: " + str(variance_of_no_shifts))

        # assert variance_of_optimal_shifts < variance_of_no_shifts

        fig, axs = plt.subplots(2, 1, dpi=200, sharex=True, figsize=(10, 10))
        ax1 = axs[0]
        ax2 = axs[1]
        # ax3 = axs[2]

        ax1.set_title(str(motif_set_length - 1) + " instances", weight='bold')
        # ax1.set_title(str("20 instances"))
        ax2.set_title("TsBubble", weight='bold')
        # ax3.set_title("Deviations along value and time axes")
        self.plotAllSeries_with_optimal_shifts(ax1, motif_set_value, motif_set_length, shifts_optimal)

        self.plot_eclipse_and_percent_around_dba(ax2, motif_set_value[0], dtw_h_to_optimal_d, dtw_vertical_deviation,
                                                    percent_v, percent_h_o, False)

        # ln1 = ax3.plot(np.arange(0, length_of_candidate), dtw_vertical_deviation, color='black', label='VWD', linewidth=0.3)
        # ax3_twin = ax3.twinx()
        # ax3_twin.spines['right'].set_color('purple')
        # ax3_twin.spines['right'].set_linewidth(4)
        # ln2 = ax3_twin.plot(np.arange(0, length_of_candidate), dtw_h_to_optimal_d, color='purple', label='HWD', linewidth=1)
        # # todo to remove
        # for iii in np.arange(80, 100):
        #     print(str(iii) + ":" + str(dtw_h_to_optimal_d[iii]))
        # lns = ln1 + ln2
        # labs = [l.get_label() for l in lns]
        # ax3.legend(lns, labs, loc=0)

        if lim_for_time_series is not None:
            ax1.set_ylim(lim_for_time_series[0], lim_for_time_series[1])
            ax2.set_ylim(lim_for_time_series[0], lim_for_time_series[1])
        # if lim_for_vertical_deviation is not None:
        #     ax3.set_ylim(lim_for_vertical_deviation[0], lim_for_vertical_deviation[1])

        plt.savefig(save_fig_name, bbox_inches='tight',  pad_inches = 0)

    def  plot_bubble_of_multi_dimension(self, motif_set_value_in_all_dimensions, length_of_candidate,
                                        assoc_tab_new_paths_in_all_dimensions, assoc_timeaxis_tab_new_paths,
                                        motif_set_length, shifts_optimal, lim_for_time_series=None,
                                        lim_for_vertical_deviation=None, save_fig_file_name=None):
        detect_dim = len(motif_set_value_in_all_dimensions)
        vertical_deviation_list = []
        average_list = []
        for dim in np.arange(detect_dim):
            average_list.append(motif_set_value_in_all_dimensions[dim][0])
            dtw_vertical_deviation, _ = self.get_vertical_deviation_and_percent(motif_set_value_in_all_dimensions[dim][0],
                                                                                       assoc_tab_new_paths_in_all_dimensions[dim])
            vertical_deviation_list.append(dtw_vertical_deviation)
        dtw_h_to_optimal_d, _ = \
            self.get_dtw_horizontal_deviation_with_shifts(motif_set_value_in_all_dimensions[0][0], assoc_timeaxis_tab_new_paths, shifts_optimal)

        fig, axs = plt.subplots(2, 1, dpi=200, sharex=True, figsize=(10, 10))
        ax1 = axs[0]
        ax2 = axs[1]
        # ax3 = axs[2]

        ax1.set_title(str(motif_set_length - 1) + " instances",  weight='bold')
        ax2.set_title("TsBubble",  weight='bold')
        # ax3.set_title("Deviations along value and time axes")

        color_arr = cm.brg(np.linspace(0, 1, detect_dim))

        for id in np.arange(1, motif_set_length):
            for dim in np.arange(detect_dim):
                ax1.plot(shifts_optimal[id] + np.arange(0, len(motif_set_value_in_all_dimensions[dim][id])),
                         motif_set_value_in_all_dimensions[dim][id], color=color_arr[dim], linewidth=0.3)

        self.plot_eclipse_around_dba_of_different_dimensions(ax2, average_list, dtw_h_to_optimal_d, vertical_deviation_list,False)

        # lns = []
        # for dim in np.arange(detect_dim):
        #     current_lns = ax3.plot(np.arange(0, length_of_candidate), dtw_vertical_deviation, color= color_arr[dim], label='VWD_dim_' + str(dim), linewidth=0.5)
        #     lns += current_lns
        # ax3_twin = ax3.twinx()
        # ax3_twin.spines['right'].set_color('black')
        # ax3_twin.spines['right'].set_linewidth(4)
        # ln2 = ax3_twin.plot(np.arange(0, length_of_candidate), dtw_h_to_optimal_d, color='black', label='HWD', linewidth=1)

        # lns += ln2
        # labs = [l.get_label() for l in lns]
        # ax3.legend(lns, labs, loc=0)

        if lim_for_time_series is not None:
            ax1.set_ylim(lim_for_time_series[0], lim_for_time_series[1])
            ax2.set_ylim(lim_for_time_series[0], lim_for_time_series[1])
        # if lim_for_vertical_deviation is not None:
        #     ax3.set_ylim(lim_for_vertical_deviation[0], lim_for_vertical_deviation[1])
        plt.savefig(save_fig_file_name)

    def find_all_series_with_indices(self, series, indices):
        target_series = list(map(lambda idx: series[idx], indices))
        return target_series
    def mapping(self, series, cluster_and_idx, max_iter=1, **kwargs):
        from dtaidistance import dtw_barycenter
        alignment_infos = []
        for cluster_idx in range(len(cluster_and_idx)):
            cur_series = self.find_all_series_with_indices(
                series, cluster_and_idx[cluster_idx]
            )
            #only support one dimension currently.
            series_mean = \
                dtw_barycenter.dba_loop(
                    s=cur_series, c=None, max_it= max_iter, thr=0.001, mask=None, keep_averages=False,
                    use_c=False, nb_initial_samples=int(0.1 * len(cur_series)), nb_prob_samples=None, keep_assoc_tab=False, **kwargs
                )
            new_but_ununsed_series_mean, unused_original_idx, assoctab, assoc_timeaxis_tab = \
                dtw_barycenter.dba_loop(
                    s=cur_series, c=series_mean, max_it=1, thr=0.001, mask=None, keep_averages=False,
                    use_c=False, nb_initial_samples=None, nb_prob_samples=None, keep_assoc_tab=True, **kwargs
                )
            alignment_info = AlignmentInfo(series_mean, unused_original_idx, assoctab, assoc_timeaxis_tab, cur_series)
            alignment_infos.append(alignment_info)

        return alignment_infos

    def generate_one_to_one_mapping(self, assoc_time_tabs):
        n_of_instances = len(assoc_time_tabs[0])
        length_of_average = len(assoc_time_tabs)
        one_to_one_mapping = [] * n_of_instances

        for id in np.arange(n_of_instances):
            current_mapping = []
            for time_index in np.arange(length_of_average):
                for k in assoc_time_tabs[time_index][id]:
                    current_mapping.extend([k, time_index])
            one_to_one_mapping.append(np.array(current_mapping).reshape((int(len(current_mapping) / 2), 2)))
        return one_to_one_mapping

    def plot_alignment(self, cur_series, average,  assoc_timeaxis_tabs, shifts_optimal):
        alpha = 0.6
        fixed_length = len(average)
        fig_overlayed, ax_overlayed = plt.subplots(2, 2, figsize=(9, 8))
        ax_overlayed[0, 0].set_axis_off()
        ax_overlayed[0, 1].set_axis_off()
        ax_overlayed[1, 0].set_axis_off()

        ax_overlayed[1, 1].invert_yaxis()
        ax_overlayed[1, 1].set_ylim([fixed_length, 0])
        ax_overlayed[1, 1].set_xlim([0, fixed_length])
        ax_overlayed[0, 1].set_xlim([0, fixed_length])

        ax_overlayed[1, 0].plot(-average, range(fixed_length, 0, -1))

        # ax_overlayed[1,1].set_aspect('equal')
        ax_overlayed[1, 1].sharex(ax_overlayed[0,1])
        # ax_overlayed[1,1].sharey(ax_overlayed[1, 0])

        one_to_one_mapping = self.generate_one_to_one_mapping(assoc_timeaxis_tabs)
        for i, path in enumerate(one_to_one_mapping):
            #not the average itself
            # (bm, em) = path[0][0], path[-1][0]

            ax_overlayed[0, 1].plot(np.arange(shifts_optimal[i], fixed_length + shifts_optimal[i]), cur_series[i],
                                    alpha=alpha)
            ax_overlayed[1, 1].plot(path[:, 0] + shifts_optimal[i], path[:, 1], ls='-', marker='.',
                                    markersize=1,
                                    alpha=alpha)
        plt.show()

    def plot_cloud_around_dba(self, alignments, centroid_id):
        alignment_info = alignments[0]
        series_mean = alignment_info.series_mean
        assoc_tab = alignment_info.assoc_tab
        assoc_timeaxis_tab = alignment_info.assoc_timeaxis_tab
        y_values = sum(assoc_tab[centroid_id], [])
        x_values = sum(assoc_timeaxis_tab[centroid_id], [])
        plt.plot( series_mean, color='purple')
        plt.scatter(x_values, y_values, color='grey')
        plt.scatter(centroid_id, series_mean[centroid_id], color="red")
        plt.show()