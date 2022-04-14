import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import os
import datetime

class DENSITYSCAN():
    def __init__(self):
        self.directory = "D:/OPTIMAL/Results/"

        """폴더 생성"""
        self.create_folder()

    def step_dbscan(self, data, scaler, x1, x2, x3, eps, min_samples):
        """
        :param data: 전처리가 완료된 데이터 입력(빈값이 없어야함.)
        :param method: 클러스터링 방법
        :param scaler: 스케일이 너무 차이가 많이나면 파라미터 설정이 어렵기 때문에 스케일을 동일하게 맞춤
        :param x1: 변수 1 ex) 외기 온도
        :param x2: 변수 2 ex) capacity
        :param x3: 변수 3 ex) Energy consumption
        :param eps: 반경
        :param min_samples: 반경 내 최소 샘플 개수를 하나의 클러스터로 만든다.
        :return: dbscan을 사용한 클러스터 결과
        """
        print("Original data shape: {}".format(data.shape))

        data = data.dropna(axis=1)
        print("Removing Nan columns:{}".format(data.shape))

        # Scaler 선택 및 실행
        if scaler == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler == 'RobustScaler':
            scaler = RobustScaler()
        elif scaler == 'MaxAbsScaler':
            scaler = MaxAbsScaler()
        else:
            scaler = StandardScaler()
        data[list(data.columns)] = scaler.fit_transform(data[list(data.columns)])

        # Clustering
        estimator = DBSCAN(eps=eps, min_samples=min_samples) #Method 선택
        clusters = estimator.fit_predict(data) #데이터 입력하여 예측
        data['cluster'] = clusters #클러스터 결과를 저장
        labels = estimator.labels_

        """클러스터 라벨 결과 저장"""
        clcr = pd.Series(labels).value_counts()
        countclusters = pd.DataFrame(clcr)
        countclusters.to_csv(self.directory + 'dbscan/countcluster_{}_{}.csv'.format(str(min_samples), str(eps*100)))

        if min_samples > 2:
            k = min_samples
        else:
            k = 2
        nbrs =  NearestNeighbors(n_neighbors=k).fit(data)
        distances, indices = nbrs.kneighbors(data)
        print("min_samples : {}".format(min_samples))
        print("shape of distances matrix : {}".format(distances.shape) + "\n")
        for enum, row in enumerate(distances[:5]):
            print("observation : {} - {}".format(str(enum), str([round(x, 2) for x in row])))
        data['knn_farthest_dist'] = distances[:, -1]
        print(data.head())
        data.to_csv(self.directory + 'dbscan/data_dbscan_{}_{}.csv'.format(str(min_samples), str(eps*100)))
        clusternumList = list(set(clusters))
        num_clusters = len(clusternumList)


        plt.clf() #그림 초기화화
        """KNN distance plottin"""
        plt.figure(figsize=(8, 8))
        plt.rcParams["font.family"] = "Times New Roman"
        fontsize = 15
        plt.subplot(1, 1, 1)
        data.sort_values('knn_farthest_dist', ascending=False).reset_index()[['knn_farthest_dist']].plot()
        plt.title("Distance(KNN)", fontsize=fontsize)
        plt.xlabel("Index", fontsize=fontsize)
        plt.ylabel("Distance", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        plt.savefig(self.directory + "dbscan/Distance_{}_{}.png".format(str(min_samples), str(eps*100)))
        plt.clf()

        """dbscan Plotting"""
        plt.figure(figsize=(8, 8))
        plt.rcParams["font.family"] = "Times New Roman"
        fontsize = 15
        plt.subplot(1, 1, 1)
        plt.scatter(x=data[x1], y=data[x2], c=data.cluster, edgecolors='black')
        plt.title("DBSCAN(clusters = {})".format(str(num_clusters)), fontsize=fontsize)
        plt.xlabel(x1, fontsize=fontsize)
        plt.ylabel(x2, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        plt.savefig(self.directory + 'dbscan/fig_dbscan_{}_{}.png'.format(str(min_samples), str(eps*100)))
        plt.clf()

        """3D scatter"""
        fig = plt.figure(figsize=(8, 8))
        plt.rcParams["font.family"] = "Times New Roman"
        fontsize = 15
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs=data[x1], ys=data[x2], zs=data[x3], c=data.cluster, edgecolors='black')
        plt.title("DBSCAN(clusters = {})".format(str(num_clusters)), fontsize=fontsize)
        plt.xlabel(x1, fontsize=fontsize)
        plt.ylabel(x2, fontsize=fontsize)
        # plt.zlabel(x3, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.zticks(fontsize=fontsize)
        plt.grid()
        plt.savefig(self.directory + 'dbscan/fig_dbscan3_{}_{}.png'.format(str(min_samples), str(eps * 100)))
        plt.clf()
        return data

    def create_folder(self):
        try:
            if not os.path.exists(self.directory +'dbscan'):
                os.makedirs(self.directory + 'dbscan')
        except OSError:
            print('Error: Creating directory. ' + self.directory + 'dbscan')



"""사용할 데이터 경로 및 파일 호출"""
today = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
folder_name = today[0:10]
path = 'D:/OPTIMAL/Data/{}/outdoor_909.csv'.format(folder_name)
data = pd.read_csv(path)

"""사용할 변수 설정"""
x1 = 'outdoor_temperature'
x2 = 'total_indoor_capa'
x3 = 'value'

# gbdata = data[['Date/Time', x1, x2]].set_index('Date/Time') #변수 2개
gbdata = data[['updated_time', x1, x2, x3]].set_index('updated_time') #변수 3개

# Scaler option : StandardScaler/MinMaxScaler/RobustScaler/MaxAbsScaler
scaler = 'MinMaxScaler'

eps = [0.1, 0.2, 0.3, 0.4]
min_samples = [100, 200, 300, 400, 500]
# min_samples = [400, 500, 600, 700, 800, 900]

for i in min_samples:
    for j in eps:
        DENSITYSCAN().step_dbscan(data=gbdata, scaler=scaler, x1=x1, x2=x2, x3=x3, eps=j, min_samples=i)