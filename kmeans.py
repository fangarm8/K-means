from ext import euc_min_dist

import numpy as np

class KMeans:
    def __init__(self, feature_num: int, cluster_num: int):
        self.feature_nums = feature_num
        self.cluster_num = cluster_num
        self.centers = []
        self.cluster_list = [[] for _ in range(self.feature_nums)]

    def init_centers(self, data):
        self.centers = []
        self.centers.append(data[np.random.randint(0, len(data))])  # Первый центроид

        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        self.centers = []
        for i in range(self.cluster_num):
            x = np.random.uniform(min_val[0], max_val[0])
            y = np.random.uniform(min_val[1], max_val[1])
            self.centers.append([x,y])

    def _update_centers_(self):
        for cluster_id, cluster in enumerate(self.cluster_list):
            cluster_mean_value = [0 for _ in range(self.feature_nums)]
            if len(cluster) != 0:
                for point in cluster:
                    for i, feature in enumerate(point):
                        cluster_mean_value[i] += feature

                for i in range(self.feature_nums):
                    cluster_mean_value[i] = cluster_mean_value[i] / len(cluster)
                self.centers[cluster_id] = cluster_mean_value

    def fit(self, data, epochs, tol=1e-4):
        labels = []
        prev_centers = None

        for _ in range(epochs):
            self.cluster_list = [[] for _ in range(self.cluster_num)]
            labels = []

            for point in data:
                min_id = euc_min_dist(point, self.centers)
                self.cluster_list[min_id].append(point)
                labels.append(min_id)
            self._update_centers_()

            if prev_centers is not None:
                if np.linalg.norm(prev_centers - np.array(self.centers)) < tol:
                    break

        return self.cluster_list, labels