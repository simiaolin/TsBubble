import os
import numpy as np
from old_uni_util import ExplanationBase
import pandas as pd

DATA_PATH = '/Users/ary/PycharmProjects/electricityPreproccessing/'
WEEK_COMSUMPTION_PATH= DATA_PATH + 'data/per_week/READING_2016_Consumption_per_week.pkl'
# WEEK_INJECTION_PATH = DATA_PATH + 'data/per_week/READING_2016_Injection_per_week.pkl'
# WEEK_OFFTAKE_PATH = DATA_PATH + 'data/per_week/READING_2016_Offtake_per_week.pkl'
DAY_COMSUMPTION_PATH = DATA_PATH + 'data/per_day/READING_2016_Consumption_per_day.pkl'
# DAY_INJECTION_PATH = DATA_PATH + 'data/per_day/READING_2016_Injection_per_day.pkl'
# DAY_OFFTAKE_PATH = DATA_PATH + 'data/per_day/READING_2016_Offtake_per_day.pkl'
data=pd.read_pickle(DAY_COMSUMPTION_PATH)
data.dropna(axis = 0, how = 'any', inplace = True)
data_transposed = data


index = data_transposed.index
columns = data_transposed.columns
first_person_info = data.loc['+EpBeN+/Wl7Osw']
# first_person_in_specific_date = first_person_info.loc[['date' == pd.to_datetime('2016-01-24')]]
first_person_in_specific_date = first_person_info.query("date" == pd.to_datetime('2016-01-24'))
for item in data_transposed.items():
    item
series = data.iloc[:30, 0:].values

print("=======")



class ExplainKmeans(ExplanationBase):

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
        predict_labels = np.zeros(series_n)
        for i in range(k):
            predict_labels[np.array(list(cluster_and_idx[i]))] = i
        return predict_labels


if __name__ == '__main__':

    k = 1
    max_dba_it = 5  # max iteration of dba
    print("series shape = " + str(series.shape[1]))

    start = 0
    end = series.shape[1]

    cluster_and_idx = {}
    cluster_and_idx[0] = set(range(0, series.shape[0]))
    explain_kmeans = ExplainKmeans(series, (start, end), max_dba_it, k)
    explain_kmeans.explain_result(1)
