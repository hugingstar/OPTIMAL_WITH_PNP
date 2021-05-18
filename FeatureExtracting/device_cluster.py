import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, complete
from sklearn.metrics.cluster import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def Clustering_hierarchy(Dic1, Dic2, data, scaler, x1, x2):
    # 계층방식의 클러스터링
    #Nomalizer
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'min_max':
        scaler = MinMaxScaler()
    elif scaler == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    col = [x1, x2]
    data[col] = scaler.fit_transform(data[col])
    print(data)

    """
    single : 최단연결법
    average : 평균연결법
    complete : 최장연결법
    """
    modes = ['single', 'average', 'complete']
    plt.figure(figsize=(40, 10))
    plt.rcParams["font.family"] = "Times New Roman"

    fontsize = 18

    y_axis = None
    for i, mode in enumerate(modes):
        y_axis = plt.subplot(1, 3, i + 1, sharey=y_axis)

        plt.title("Hierarchy, linkage mode : {}".format(mode), fontsize=fontsize)
        plt.xlabel('Distance', fontsize=fontsize)
        plt.ylabel('Device', fontsize=fontsize)
        clustering = linkage(data[[x1, x2]], mode)
        print("{} : {}".format(mode, clustering))

        dendrogram(clustering, labels=data['Indoor_unit_num'].tolist(), orientation='right')
        plt.xticks(fontsize=fontsize - 2)
        plt.yticks(fontsize=fontsize - 2)
        plt.grid()
    plt.tight_layout()
    plt.savefig(Dic2 + '/fig_dendrogram.png')
    plt.clf()
    return 0


def Clustering_KMeans_K(Dic1, Dic2, data, scaler, n_clusters, x1, x2):
    # KMEAN 방식의 클러스터링

    # Nomalizer
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'min_max':
        scaler = MinMaxScaler()
    elif scaler == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    print(data.columns)
    print(data.index)
    data[list(data.columns)] = scaler.fit_transform(data[list(data.columns)])

    plt.figure(figsize=(6, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    fontsize = 15
    cost = []
    # Clustering
    estimator = KMeans(n_clusters=n_clusters, random_state=0)
    etm = estimator.fit(data)

    """silhoutte score"""
    silhoutte = etm.labels_
    siscore = [float(silhouette_score(data, silhoutte))]
    print("Silhouette Score:", siscore)
    sil = pd.DataFrame()
    sil['silhouette_score'] = siscore
    sil.to_csv(Dic2 + "/silhouette_score.csv")

    cost.append(abs(estimator.score(data)))
    cluster_ids = estimator.fit_predict(data)
    print("cluster ids : {}".format(cluster_ids))
    clustered_frame = pd.DataFrame({"Indoor_unit_num": data.index.to_list(), "Cluster": cluster_ids})
    clustered_frame.to_csv(Dic2 + "/Ind_kmeans_K_{}.csv".format(n_clusters))
    print("Cluster centers : {}".format(estimator.cluster_centers_))

    """2D scatter"""
    plt.subplot(1, 1, 1)
    plt.title("KMeans, K = {}".format(n_clusters), fontsize=fontsize)

    plt.xlabel(x1, fontsize=fontsize)
    plt.ylabel(x2, fontsize=fontsize)
    plt.scatter(data[x1], data[x2], c=cluster_ids)
    # for name, mark, attended in data.itertuples():
    #     plt.annotate(name, (attended, mark))
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig(Dic2 + '/fig_kmeans_k_{}.png'.format(n_clusters))
    plt.clf()

    plt.figure(figsize=(15, 5))
    plt.stem(range(1, len(cost) + 1), cost)
    plt.title("KMeans scores", fontsize=fontsize)
    plt.xlabel("K-values", fontsize=fontsize)
    plt.ylabel("Score", fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.grid(alpha=0.3)
    plt.savefig(Dic2 + '/fig_kmeans_score_k_{}.png'.format(n_clusters))
    plt.clf()
    # plt.show()
    return

# def Clustering_KMeans(Dic1, Dic2, data, scaler, x1, x2):
#     #Nomalizer
#     if scaler == 'standard':
#         scaler = StandardScaler()
#     elif scaler == 'min_max':
#         scaler = MinMaxScaler()
#     elif scaler == 'robust':
#         scaler = RobustScaler()
#     else:
#         scaler = StandardScaler()
#
#     # print(data.columns)
#     data[list(data.columns)] = scaler.fit_transform(data[list(data.columns)])
#     # print(data)
#
#     plt.figure(figsize=(15, 10))
#     plt.rcParams["font.family"] = "Times New Roman"
#     fontsize = 15
#
#     cost = []
#     for i in range(3, 9):
#         # Clustering
#         estimator = KMeans(n_clusters=i, random_state=0)
#         estimator.fit(data)
#         cost.append(abs(estimator.score(data)))
#         cluster_ids = estimator.fit_predict(data)
#         clustered_frame = pd.DataFrame({"Indoor_unit_num": data.index.to_list(), "Cluster": cluster_ids})
#         clustered_frame.to_csv(Dic2 + "/Ind_kmeans_K_{}.csv".format(i))
#         print(cluster_ids)
#
#         plt.subplot(2, 3, i - 2)
#         plt.title("KMeans, K = {}".format(i), fontsize=fontsize)
#         plt.xlabel(x1, fontsize=fontsize)
#         plt.ylabel(x2, fontsize=fontsize)
#         plt.scatter(data[x1], data[x2], c=cluster_ids)
#         # for name, mark, attended in data.itertuples():
#         #     plt.annotate(name, (attended, mark))
#         plt.tight_layout()
#         plt.grid(alpha=0.3)
#         plt.savefig(Dic2 + '/fig_kmeans.png')
#     plt.clf()
#     plt.figure(figsize=(15, 5))
#     plt.stem(range(1, len(cost) + 1), cost)
#     plt.title("KMeans scores", fontsize=fontsize)
#     plt.xlabel("K-values", fontsize=fontsize)
#     plt.ylabel("Score", fontsize=fontsize)
#     plt.xticks(fontsize=fontsize - 2)
#     plt.yticks(fontsize=fontsize - 2)
#     plt.grid(alpha=0.3)
#     plt.savefig(Dic2 + '/fig_kmeans_score.png')
#     # plt.show()
#     return


Dic1 = './Data/'
Dic2 = './Results/'

"""Features"""
x0 = 'Indoor_unit_num'
x1 = 'Indoor_heating_capacity'
x2 = 'Zone_area'
n_clusters = 4

scaler ='min_max' #min_max, standard, robust

df = pd.read_csv(Dic1 + "system_data2.csv")
gbdata = df[[x0, x1, x2]]
gbdata = gbdata.set_index(x0)
# print(gb_data)

# Clustering_hierarchy(Dic1=Dic1, Dic2=Dic2, data=gb_data, scaler=scaler, x1=x1, x2=x2)
Clustering_KMeans_K(Dic1=Dic1, Dic2=Dic2, data=gbdata, n_clusters=n_clusters, scaler=scaler, x1=x1, x2=x2)
# Clustering_KMeans(Dic1=Dic1, Dic2=Dic2, data=gbdata, scaler=scaler, x1=x1, x2=x2)