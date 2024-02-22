# coding: utf-8

import numpy as np
import pandas as pd
from kmeans_dba import ExplainKmeans
import sys
import matplotlib.pyplot as plt
from tsbubble.ts_bubble import  TsBubble

class MotifDiscovery:

    def datapreprocessing(self):
        ...
    def mapping_preparing(self, series, induced_paths, n_of_timeseries, representative_size, b):
        dim = series.shape[1]
        assoc_tabs = [[[[] for _ in range(n_of_timeseries)] for _ in range(representative_size)] for _ in range(dim)]

        # It is the same for different dimensions
        assoc_timeaxis_tabs = [[[] for _ in range(n_of_timeseries)] for _ in range(representative_size)]
        series_flip = series.transpose().reshape(-1)

        for i, path in enumerate(induced_paths):
            (bm, em) = path[0][0], path[-1][0]
            for mapping in path:
                assoc_timeaxis_tabs[mapping[1] - b][i].append(mapping[0] - bm)
                for dim in np.arange(series.shape[1]):
                    assoc_tabs[dim][mapping[1] - b][i].append(series_flip[mapping[0] + dim * series.shape[0]])

        return assoc_tabs, assoc_timeaxis_tabs
