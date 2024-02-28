import pickle
path = "/Users/ary/PycharmProjects/TsBubble/affinities_permanent/affinity_matrix/CBF.pkl"
path2 = "/Users/ary/PycharmProjects/cobras/examples/affinities_permanent/CBF.pkl"
path = "/Users/ary/PycharmProjects/cobras/examples/affinities_permanent/Trace.pkl"
path2 = "/Users/ary/PycharmProjects/cobras/examples/clustering_result/Trace/50.pkl"
path3 = "/Users/ary/PycharmProjects/TsBubble/affinities_permanent/clustering_result/Trace/50.pkl"

from cobras_ts.clustering import Clustering
def show_clustering_info (clustering: Clustering):
   for cls in clustering.clusters:
       for si in cls.super_instances:
           print("repre " + str(si.representative_idx))
           print("instance " + str(si.indices))

fileread = open(path2, 'rb')
cls_1 = pickle.load(fileread)
show_clustering_info(cls_1)
fileread.close()
print(" ----- ")
fileread = open(path3, 'rb')
cls_2 = pickle.load(fileread)
show_clustering_info(cls_2)
fileread.close()

