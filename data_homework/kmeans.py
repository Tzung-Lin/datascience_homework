import random

from matplotlib import pyplot as plt
import numpy as np


# 返回一个分类好的字典,key 为质心在质心数组中的下标，value 为属于该质心的坐标
def classify_cluster(dataset, centroid_arr):
    cluster_dict = dict()
    for point in dataset:
        nearest_centroid_index = -1
        min_distance = float('inf')
        for i in range(len(centroid_arr)):
            if cal_euclidean_distance(point, centroid_arr[i]) < min_distance:
                min_distance = cal_euclidean_distance(point, centroid_arr[i])
                nearest_centroid_index = i
        if nearest_centroid_index not in list(cluster_dict.keys()):
            cluster_dict[nearest_centroid_index] = []
        cluster_dict[nearest_centroid_index].append(point)
    #print('分类结果：',cluster_dict)
    return cluster_dict


# 返回新的质心
def choose_new_centroid(cluster_arr):
    result = [[] * len(cluster_arr)]
    print(len(cluster_arr))
    for i in range(len(cluster_arr)):
        print(i,np.mean(cluster_arr[i],axis=0))
        result.insert(i, np.mean(cluster_arr[i],axis=0))
    #print('質心數組：',result)
    result.pop()
    #print('result:',result)
    return result


# 生成不重复的 k 个 index,并返回质心数组
def random_choose_centroid(dataset, k):
    rng = np.random.default_rng()
    index_arr = rng.choice(a=len(dataset), size=k, replace=False)
    centroid_arr = [dataset[idx] for idx in index_arr]
    return centroid_arr


def cal_euclidean_distance(point1, point2):
    #print('计算欧式距离：',point1,point2)
    return np.sqrt(np.sum(np.square(point1 - point2)))


def show_cluster(centroids, cluster_dict):
    """
    展示聚类结果
    :param centroids:
    :param cluster_dict:
    :return:
    """
    color_mark = ['or', 'ob', 'og', 'ok', 'oy', 'om', 'oc']
    centroid_mark = ['dr', 'db', 'dg', 'dk', 'dy', 'dm', 'dc']
    for key in cluster_dict.keys():
        # 画质心
        plt.plot(centroids[key][0], centroids[key][1], centroid_mark[key], markersize=12)  # 质心点
        for node in cluster_dict[key]:
            # 画顶点
            plt.plot(node[0], node[1], color_mark[key])
            #plt.text(node[0], node[1], '1', ha='center', va='bottom', fontsize=10.5)
    plt.xlabel('1/upload_times')
    plt.ylabel('total score')
    plt.show()

# 计算所有顶点和其质心的距离和作为损失函数
def get_variance(centroids, cluster_dict):
    """
    计算各个簇集合的均方误差
    将簇类中各个节点与质心的距离累加求和
    :param centroids:
    :param cluster_dict:
    :return:
    """
    sum = 0.0
    for cluster_idx in cluster_dict.keys():
        centroid = centroids[cluster_idx]
        distance = 0.0
        for node in cluster_dict[cluster_idx]:
            distance += cal_euclidean_distance(node, centroid)
        sum += distance
    return sum


