from dtaidistance import dtw_barycenter
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.text  import Text
import matplotlib.pyplot as plt
import bisect
plt.rcParams.update({'font.size': 5})

class AlignmentInfo:
    def __init__(self, series_mean, original_idx, assoc_tab, assoc_timeaxis_tab, cur_series):
        self.series_mean = series_mean
        self.original_idx = original_idx
        self.assoc_tab = assoc_tab
        self.assoc_timeaxis_tab = assoc_timeaxis_tab
        self.cur_series = cur_series
class ExplanationBase:

    def __init__(self, series, span, max_iter, k, align_info_provided = False):
        self.series = series
        self.x = np.arange(span[0], span[1])
        self.dba_max_iter = max_iter
        self.span = span
        self.k = k
        if not align_info_provided:
            self.alignment_infos = self.get_alignment_infos(k)

    def init_figure(fig):

        # ax1 = fig.add_subplot(231)  # DBA
        ax2 = fig.add_subplot(411)  # adjusted dots
        ax3 = fig.add_subplot(412, sharex=ax2, sharey=ax2)  # eclipse
        ax5 = fig.add_subplot(413, sharex = ax2, sharey = ax2)
        ax4 = fig.add_subplot(414, sharex=ax2)  # dtw_vertical
        # ax5 = fig.add_subplot(235, sharex=ax1, sharey=ax1)  # vertical
        # ax6 = fig.add_subplot(236, sharex=ax1, sharey=ax1)  # dtw_horizontal

        ax_overall = [ax2, ax3, ax5]
        ax_statistic = [ax4]
        axs = ax_overall + ax_statistic

        # axis color
        for ax in axs:
            ax.spines['bottom'].set_color('blue')
            # ax.spines['bottom'].set_linewidth(1)
            # ax.spines['left'].set_linewidth(1)
            # ax.set_xlabel("time index")

        for ax in ax_overall:
            ax.spines['left'].set_color('pink')
        for ax in ax_statistic:
            ax.spines['left'].set_color('yellow')
        ax4.set_xlabel("time index")
        # ax1.set_ylabel("DBA", color="purple")
        ax2.set_title("Instances", color="blue")
        ax3.set_title("Small deviation Bubbles", color="blue", size = 5)
        ax5.set_title("Big deviation Bubbles", color="blue", size = 5)
        # ax6.set_ylabel("DTW Horizontal Deviation", color="green")
        ax4.set_title("vertical warping deviation VS horizontal warping deviation", color="red", size = 5)
        # ax5.set_ylabel("EU Deviation", color="gray")
        return ax2, ax3, ax4, ax5


    def init_figure_for_different_deviations():

        fig, axs = plt.subplots(6, 1, dpi=200, sharex=True, figsize=(10, 10))
        ax1 = axs[0]
        ax2 = axs[1]
        ax3 = axs[2]
        ax4 = axs[3]
        ax5 = axs[4]
        ax6 = axs[5]

        # plt.subplots_adjust()
        # ax1 = fig.add_subplot(231)  # DBA
        # ax1 = fig.add_subplot(611)  # adjusted dots
        # ax2 = fig.add_subplot(612, sharex=ax1)  #to left
        # ax3 = fig.add_subplot(613, sharex = ax1) # to right
        # ax4 = fig.add_subplot(614, sharex=ax1)  # to middle
        # ax5 = fig.add_subplot(615, sharex = ax1) # to optimal
        # ax6 = fig.add_subplot(616, sharex = ax1) # all the deviations

        ax_overall = [ax1, ax2, ax3]
        ax_statistic = [ax4]
        axs = ax_overall + ax_statistic

        # axis color
        # for ax in axs:
        #     ax.spines['bottom'].set_color('blue')

        # for ax in ax_overall:
        #     ax.spines['left'].set_color('pink')¡
        # for ax in ax_statistic:
        #     ax.spines['left'].set_color('yellow')
        # ax4.set_xlabel("time index")
        # ax1.set_ylabel("DBA", color="purple")
        ax1.set_title("All Instances", color="blue")
        ax2.set_title("Bubbles (warping to left)", color="blue", size = 5)
        ax3.set_title("Bubbles (warping to right)", color="blue", size = 5)
        ax4.set_title("Bubbles (warping to middle", color = 'blue', size = 5)
        ax5.set_title("Bubbles (warping to optimal)", color = 'blue', size = 5)
        ax6.set_title("All types of deviations", color="red", size = 5)
        # ax5.set_ylabel("EU Deviation", color="gray")
        return ax1, ax2, ax3, ax4,ax5, ax6


    def find_all_series_with_indices(self, series, indices):
        target_series = list(map(lambda idx: series[idx], indices))
        return target_series

    def get_cluster_sequences_dict(self):
        pass

    def get_color_iter(self, size):
        color_arr = cm.rainbow(np.linspace(0, 1, size))
        colors = iter(color_arr)
        return colors

    def plot_dba(self, plt, series_mean, vertical_deviation):
        if self.span is None:
            plt.plot(self.x, series_mean, color='purple',linewidth = 0.3)
            # plt.fill_between(
            #     self.x[self.span[0]: self.span[1]],
            #     (series_mean + 3 * vertical_deviation)[self.span[0]: self.span[1]],
            #     (series_mean - 3 * vertical_deviation)[self.span[0]: self.span[1]],
            #     facecolor='grey',
            #     edgecolor='orange',
            #     alpha=0.3
            #     )
        elif self.span[0] + 1 == self.span[1]:
            plt.scatter(self.x[self.span[0]], series_mean[self.span[0]],  color = 'purple')
        else:
            plt.plot(self.x[self.span[0]: self.span[1]], series_mean[self.span[0]: self.span[1]], color = 'purple' ,linewidth = 0.3)
            # plt.fill_between(self.x[self.span[0]: self.span[1]],
            #                  (series_mean + 3 * vertical_deviation)[self.span[0]: self.span[1]],
            #                  (series_mean - 3 * vertical_deviation)[self.span[0]: self.span[1]],
            #                  facecolor='grey',
            #                  edgecolor='orange',
            #                  linewidth=0.2,
            #                  alpha = 0.3)

    def plot_eclipse_around_dba(self, plt, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation, order):  #plotting elipses without overlapping among them.
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
    def plot_eclipse_and_percent_around_dba(self, plt, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation, v_percent, h_percent, order):  #plotting elipses without overlapping among them.
        plt.plot(series_mean, color='purple' ,linewidth = 0.3)
        color_type_num = 10
        color_arr = cm.rainbow(np.linspace(0, 1, color_type_num))
        elipse_index_list = self.get_elipse_index_list(hwd=dtw_horizontal_deviation, vwd=dtw_vertical_deviation, order=order)
        ells = [Ellipse(xy=(i, series_mean[i]),
                        width=dtw_horizontal_deviation[i], height=dtw_vertical_deviation[i], color=color_arr[i%color_type_num])
                for i in elipse_index_list
                ]

        textlist = [Text(i, series_mean[i],
                         text  = 'h:'+"%.2f" % h_percent[i] + '\nv:' + "%.2f"%v_percent[i])
                    for i in elipse_index_list]
        for e in ells:
            plt.add_artist(e)
        for t in textlist:
            plt.add_artist(t)


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

    def get_dtw_vertical_deviation(self, series_mean, assoc_tab):   ##should use the old mean or the new mean?
        dtw_vertical_deviation = np.zeros(shape=series_mean.shape, dtype=np.double)
        for i in range(len(series_mean)):
            all_values_aligned_to_current_idx = []
            for _, values in enumerate(assoc_tab[i]):
                all_values_aligned_to_current_idx = np.append(all_values_aligned_to_current_idx, values)
            dtw_vertical_deviation[i] = np.sqrt(np.divide(
                np.sum(np.square(all_values_aligned_to_current_idx - series_mean[i])),
                len(all_values_aligned_to_current_idx)
                ))
        return dtw_vertical_deviation
    def get_dtw_vertical_deviation_and_percent(self, series_mean, assoc_tab):   ##should use the old mean or the new mean?
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

    def get_eu_deviation(self, series_mean, cur_series, original_idx):
        marked_series = []
        for i in original_idx:
            marked_series.append(cur_series[i])
        return np.sqrt(np.divide(
            np.sum(np.square(marked_series - series_mean), axis=0),
            len(original_idx)
            ))

    def get_dtw_horizontal_deviation(self, series_mean, assoc_timeaxis_tab): ##should use the old mean or the new mean?
        dtw_horizontal_deviation = np.empty(shape=series_mean.shape)
        for i in range(len(series_mean)):
            all_timeaxis_assigned_to_current_idx = []
            for _, idxes in enumerate(assoc_timeaxis_tab[i]):
                all_timeaxis_assigned_to_current_idx = np.append(all_timeaxis_assigned_to_current_idx, idxes)
            dtw_horizontal_deviation[i] = np.sqrt(np.divide(
                np.sum(np.square(all_timeaxis_assigned_to_current_idx - i)),
                len(all_timeaxis_assigned_to_current_idx)
                ))
        return dtw_horizontal_deviation

    def get_dtw_horizontal_deviation_with_shifts(self, series_mean, assoc_timeaxis_tab, shifts):  #get the horizontal deviation with a shift
        dtw_horizontal_deviation = np.empty(shape=series_mean.shape)
        inside_one_deviation_percentage = np.empty(shape=series_mean.shape)
        for i in range(len(series_mean)):
            all_timeaxis_assigned_to_current_idx = []
            for j, idxes in enumerate(assoc_timeaxis_tab[i]):
                all_timeaxis_assigned_to_current_idx = np.append(all_timeaxis_assigned_to_current_idx, idxes + shifts[j])
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

    def plot_adjusted_series(
            self, plt, cluster_id, assoc_timeaxis_tab, original_idx, series
            ):  ##only plot series that are mapped to the range
        size_of_sequence = len(original_idx)
        plt.set_title(str(size_of_sequence) + " instances of cluster " + str(cluster_id), size = 5)
        colors = self.get_color_iter(size_of_sequence)  ## the size of all the series that are used to calculate the mean

        for new_idx in range(size_of_sequence):
            s = assoc_timeaxis_tab[self.span[0]][new_idx][0]
            e = assoc_timeaxis_tab[self.span[1] - 1][new_idx][-1]
            x = np.arange(s, e)
            original_series_idx = original_idx[new_idx]
            y = list(map(lambda x: series[original_series_idx][x], x))
            if len(x) == 1:
                plt.scatter(x, y, color=next(colors),linewidth = 0.3)
            else:

                if new_idx == 0: # the centroid of the local motif case
                    plt.plot(x, y, color = 'purple', linewidth = 2)
                else:
                    plt.plot(x, y, color=next(colors) ,linewidth = 0.3)
            # expected_range = np.arange(range[0], range[1])
            # non_plotted_range = [item for item in expected_range if item not in x]
            # for non_plotted_dot in non_plotted_range:
            #     plt.scatter(non_plotted_dot, series[new_idx][non_plotted_dot], color = 'black')

    def plotAllSeries(self, plt,  motif_set_value, size):
        colors = self.get_color_iter(size)  ## the size of all the series that are used to calculate the mean

        for new_idx in range(size):
            x = np.arange(0, len(motif_set_value[new_idx]))

            if new_idx == 0: # the centroid of the local motif case
                plt.plot(x, motif_set_value[new_idx], color = 'purple', linewidth = 2)
            else:
                plt.plot(x, motif_set_value[new_idx], color=next(colors) ,linewidth = 0.3)

    def plotAllSeries_with_optimal_shifts(self, plt,  motif_set_value, size, shifts_optimal):
        colors = self.get_color_iter(size)  ## the size of all the series that are used to calculate the mean

        for new_idx in range(size):
            if new_idx == 0: # the centroid of the local motif case
                x = np.arange(0, len(motif_set_value[new_idx]))
                plt.plot(x, motif_set_value[new_idx], color = 'purple', linewidth = 2)
            else:
                plt.plot(np.arange(shifts_optimal[new_idx], len(motif_set_value[new_idx]) + shifts_optimal[new_idx]), motif_set_value[new_idx], color=next(colors) ,linewidth = 0.3)
    def find_all_indices_of_one_cluster(self, cluster_idx):
        pass

    def get_alignment_infos(self, cluster_size):
        alignment_infos = []
        for cluster_idx in range(cluster_size):
            cur_series = self.find_all_series_with_indices(
                self.series, self.find_all_indices_of_one_cluster(cluster_idx)
                )
            series_mean, original_idx, assoctab, assoc_timeaxis_tab = \
                dtw_barycenter.dba_loop(
                    s=cur_series, c=None, max_it=self.dba_max_iter, thr=0.001, mask=None, keep_averages=False,
                    use_c=False, nb_initial_samples=None, nb_prob_samples=None, keep_assoc_tab=True
                    )
            alignment_info = AlignmentInfo(series_mean, original_idx, assoctab, assoc_timeaxis_tab, cur_series)
            alignment_infos.append(alignment_info)
        return alignment_infos
    def plot_selected_span(self, cluster_idx,  ax2, ax3, ax4, ax5):
        alignment_info = self.alignment_infos[cluster_idx]
        series_mean, original_idx, assoc_tab, assoc_timeaxis_tab, cur_series =\
            alignment_info.series_mean, alignment_info.original_idx, alignment_info.assoc_tab, alignment_info.assoc_timeaxis_tab, alignment_info.cur_series
        dtw_vertical_deviation = self.get_dtw_vertical_deviation(series_mean, assoc_tab)
        dtw_horizontal_deviation = self.get_dtw_horizontal_deviation(series_mean, assoc_timeaxis_tab)
        eu_vertical_deviation = self.get_eu_deviation(series_mean, cur_series, original_idx)

        # self.plot_dba(ax1, series_mean, dtw_vertical_deviation)
        self.plot_adjusted_series(ax2, cluster_idx, assoc_timeaxis_tab, original_idx, cur_series)
        self.plot_statistics_curves(ax4, dtw_vertical_deviation, eu_vertical_deviation, dtw_horizontal_deviation)
        self.plot_eclipse_around_dba(ax3, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation, True)
        self.plot_eclipse_around_dba(ax5, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation, False)
        # self.plot_cloud_around_dba(ax5, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation)
        # self.plot_single_eclipe_around_dba(ax6, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation)
    def plot_cloud_around_dba(self, cluster_id, centroid_id):
        alignment_info = self.alignment_infos[cluster_id]
        series_mean = alignment_info.series_mean
        assoc_tab = alignment_info.assoc_tab
        assoc_timeaxis_tab = alignment_info.assoc_timeaxis_tab
        y_values = sum(assoc_tab[centroid_id], [])
        x_values = sum(assoc_timeaxis_tab[centroid_id], [])
        plt.plot(self.x, series_mean, color='purple')
        plt.scatter(x_values, y_values, color='grey')
        plt.scatter(centroid_id, series_mean[centroid_id], color="red")

    def plot_deviations(self, plt, values, color, deviation_label):
        if self.span is None:
            plt.plot(self.x, values, color=color, label = deviation_label ,linewidth = 0.3)
        elif self.span[0] + 1 == self.span[1]:
            plt.scatter(self.x[self.span[0]], values[self.span[0]], color=color, label = deviation_label)
        else:
            plt.plot(self.x[self.span[0]: self.span[1]], values[self.span[0]: self.span[1]], color=color, label = deviation_label ,linewidth = 0.3)

    def plot_statistics_curves(self, ax4, dtw_vertical_deviation, eu_vertical_deviation, dtw_horizontal_deviation):
        self.plot_deviations(ax4, dtw_vertical_deviation, 'red', "VWD")
        self.plot_deviations(ax4, eu_vertical_deviation, 'grey', "STD")
        self.plot_deviations(ax4, dtw_horizontal_deviation, 'green', "HWD")

    def plot_a_cluster(self, i):
        fig = plt.figure(i+1, dpi=300)
        ax2, ax3, ax4, ax5 = ExplanationBase.init_figure(fig)
        self.plot_selected_span(i,  ax2, ax3, ax4, ax5)

    def explain_result(self, cluster_size):
        for i in range(cluster_size):
            self.plot_a_cluster(i)
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.show()

    def plot_points_aligned_to_a_given_centroid_index(self,  cluster_id, centroid_id):

        alignment_info = self.alignment_infos[cluster_id]
        series_mean = alignment_info.series_mean
        assoc_tab = alignment_info.assoc_tab
        assoc_timeaxis_tab = alignment_info.assoc_timeaxis_tab
        y_values = sum(assoc_tab[centroid_id], [])
        x_values = sum(assoc_timeaxis_tab[centroid_id], [])
        plt.plot(self.x, series_mean, color='purple')
        plt.scatter(x_values, y_values, color = 'grey')
        plt.scatter(centroid_id, series_mean[centroid_id], color = "red")

    def plot_single_bubble_around_dba(self, plt, cluster_id, centroid_id):
        alignment_info = self.alignment_infos[cluster_id]
        series_mean = alignment_info.series_mean
        assoc_tab = alignment_info.assoc_tab
        assoc_timeaxis_tab = alignment_info.assoc_timeaxis_tab
        dtw_vertical_deviation = self.get_dtw_vertical_deviation(series_mean, assoc_tab)
        dtw_horizontal_deviation = self.get_dtw_horizontal_deviation(series_mean, assoc_timeaxis_tab)
        ax1 = plt.add_subplot(111)  # DBA

        self.plot_eclipse_around_dba_on_give_time_index(ax1, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation, centroid_id)

    def plot_eclipse_around_dba_on_give_time_index(self, plt, series_mean, dtw_horizontal_deviation, dtw_vertical_deviation, centroid_id):
        plt.plot(series_mean, color='purple')
        plt.add_artist(Ellipse(
                        xy=(centroid_id, series_mean[centroid_id]),
                        width=dtw_horizontal_deviation[centroid_id],
                        height=dtw_vertical_deviation[centroid_id],
                        color="green"))

