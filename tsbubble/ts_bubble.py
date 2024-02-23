import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.text  import Text
import matplotlib.cm as cm
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
                x = np.arange(0, len(motif_set_value[new_idx]))
                plt.plot(x, motif_set_value[new_idx], color = 'purple', linewidth = 2)
            else:
                plt.plot(np.arange(shifts_optimal[new_idx], len(motif_set_value[new_idx]) + shifts_optimal[new_idx]), motif_set_value[new_idx], color=next(colors) ,linewidth = 0.3)
    def get_elipse_index_list(self,  hwd, vwd, order): #horizontal wraping deviation, vertical wraping deviation
        n = len(hwd)
        area = list(map(lambda i: hwd[i] * vwd[i],  range(n)))
        sort_index = np.argsort(area)   # small elipses rank first
        if not order:
            sort_index = np.flip(sort_index)  #bigger elipses rank first

        #for the biggier elipses.

        occupied_left_index_list = [float('-inf'), float('inf')]
        occupied_right_index_list = [float('-inf'), float('inf')]
        elipse_index_list = []
        for i in sort_index:
            updated = self.try_insert_elipse(occupied_left_index_list, occupied_right_index_list ,new_index=i, width=hwd[i])
            if updated:
                elipse_index_list.append(i)
        return elipse_index_list

    def try_insert_elipse(self, occupied_left_index_list, occupied_right_index_list, new_index, width):
        left = new_index - width
        right = new_index + width
        index = bisect.bisect_right(occupied_left_index_list, left)

        last_right = occupied_right_index_list[index - 1]
        next_left = occupied_left_index_list[index]
        if left >= last_right and right <= next_left:
            occupied_left_index_list.insert(index, left)
            occupied_right_index_list.insert(index, right)
            return True
        else:
            return False
    def plot_eclipse_and_percent_around_dba(self, plt, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation, v_percent, h_percent, order,  percentageOn = False):  #plotting elipses without overlapping among them.
        plt.plot(series_mean, color='purple' ,linewidth = 0.3)
        color_type_num = 10
        color_arr = cm.rainbow(np.linspace(0, 1, color_type_num))
        elipse_index_list = self.get_elipse_index_list(hwd=dtw_horizontal_deviation, vwd=dtw_vertical_deviation, order=order)
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


    def plot_bubble_of_one_dimension(self, motif_set_value, length_of_candidate,
                                     assoc_tab_new_paths, assoc_timeaxis_tab_new_paths, motif_set_length, shifts_optimal):
        # explain = ExplainKmeans(series=motif_set_value, span=(0, length_of_candidate - 1), max_iter=None, k=1,
        #                         dist_matrix=None, V=None, cluster_and_idx=None, align_info_provided=True)
        dtw_vertical_deviation, percent_v = self.get_vertical_deviation_and_percent(motif_set_value[0],
                                                                                       assoc_tab_new_paths)
        dtw_h_to_optimal_d, percent_h_o = \
            self.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths, shifts_optimal)

        variance_of_optimal_shifts = np.sum(np.square(dtw_h_to_optimal_d))

        dtw_h_no_shifts, percent_h_o = \
            self.get_dtw_horizontal_deviation_with_shifts(motif_set_value[0], assoc_timeaxis_tab_new_paths,
                                                          [np.float64(0)] * len(shifts_optimal))
        variance_of_no_shifts = np.sum(np.square(dtw_h_no_shifts))
        print("the variance of optimal shifts: " + str(variance_of_optimal_shifts))

        print("the variance without shifts: " + str(variance_of_no_shifts))

        assert variance_of_optimal_shifts < variance_of_no_shifts

        fig, axs = plt.subplots(3, 1, dpi=200, sharex=True, figsize=(10, 10))
        ax1 = axs[0]
        ax2 = axs[1]
        ax3 = axs[2]

        ax1.set_title("All Instances", color="blue")
        ax2.set_title("Bubbles (warping to optimal)", color='blue', size=5)
        ax3.set_title("All types of deviations", color="red", size=5)
        self.plotAllSeries_with_optimal_shifts(ax1, motif_set_value, motif_set_length, shifts_optimal)

        self.plot_eclipse_and_percent_around_dba(ax2, motif_set_value[0], dtw_h_to_optimal_d, dtw_vertical_deviation,
                                                    percent_v, percent_h_o, False)

        ax3.plot(np.arange(0, length_of_candidate), dtw_vertical_deviation, color='black', label='VWD', linewidth=0.3)
        ax3_twin = ax3.twinx()
        ax3_twin.spines['right'].set_color('purple')
        ax3_twin.spines['right'].set_linewidth(4)
        ax3_twin.plot(np.arange(0, length_of_candidate), dtw_h_to_optimal_d, color='purple', label='HWD_optimal', linewidth=2)
        plt.legend(loc='best')
        plt.show()

    def find_all_series_with_indices(self, series, indices):
        target_series = list(map(lambda idx: series[idx], indices))
        return target_series
    def mapping(self, series, cluster_and_idx):
        from dtaidistance import dtw_barycenter
        alignment_infos = []
        for cluster_idx in range(len(cluster_and_idx)):
            cur_series = self.find_all_series_with_indices(
                series, cluster_and_idx[cluster_idx]
            )
            #only support one dimension currently.
            series_mean, original_idx, assoctab, assoc_timeaxis_tab = \
                dtw_barycenter.dba_loop(
                    s=cur_series, c=None, max_it=3, thr=0.001, mask=None, keep_averages=False,
                    use_c=False, nb_initial_samples=None, nb_prob_samples=None, keep_assoc_tab=True
                )

            alignment_info = AlignmentInfo(series_mean, original_idx, assoctab, assoc_timeaxis_tab, cur_series)
            alignment_infos.append(alignment_info)

        return alignment_infos

