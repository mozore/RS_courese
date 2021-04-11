# -- coding: utf-8 --
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_path='/data/u.data'):
    """
    加载movielens评分数据
    :param file_path: ratings数据存储位置
    :return: 评分对(uid, mid, rating)数组，用户数量，电影数量
    """
    data = []
    for line in open(file_path, 'r'):
        arr = line.split()
        uid = int(arr[0])
        mid = int(arr[1])
        rating = int(arr[2])
        data.append([uid, mid, rating])
    data = np.array(data)
    n_users = np.max(data[:, 0]) + 1
    n_movies = len(np.unique(data[:, 1])) + 1

    return np.array(data), n_users, n_movies


def split_data(data, test_size=0.2):
    """
    分割数据集为训练集和测试集
    :param data: 原始评分数据
    :param test_size: 划分的比例
    :return: 训练集和数据集array train_data, test_data
    """
    return train_test_split(data, test_size)
