# -- coding: utf-8 --
import numpy as np
import math


class PMF:
    def __init__(self, n_feat=20, lr=0.005, lam_u=0.1, lam_v=0.1, n_epoches=20):
        """

        :param n_feat: 特征数量
        :param lr: 学习率
        :param lam_u: 用户的方差
        :param lam_v: 物品方差
        :param n_epoches: 迭代轮数
        """
        self.n_feat = n_feat
        self.n_epoches = n_epoches
        self.lr = lr
        self.lam_u = lam_u
        self.lam_v = lam_v
        self.e = 0  # 迭代轮数的index
        self.U = None  # 用户矩阵
        self.V = None  # 物品矩阵
        self.n_users = 0  # 用户数量
        self.n_movies = 0  # 电影数量

    def set_num(self, n_users, n_movies):
        self.n_users = n_users
        self.n_movies = n_movies

    def fit(self, train_data, test_data):
        """

        :param train_data: 训练数据
        :param test_data: 测试数据
        :param n_users: 用户数量
        :param n_movies: 电影数量
        :return: train_rmse, test_rmse 训练集测试集的RMSE
        """
        n_train = train_data.shape[0]
        n_test = test_data.shape[0]
        if self.U is None or self.V is None:
            self.e = 0  # 初始化迭代次数
            self.U = np.random.rand(self.n_users, self.n_feat)  # 随机初始化用户向量
            self.V = np.random.rand(self.n_movies, self.n_feat)  # 随机初始化电影向量
            # print(self.U)
        # RMSE参数
        train_rmse = []
        test_rmse = []

        while self.e < self.n_epoches:
            self.e += 1
            np.random.shuffle(train_data)  # shuffle

            for data in train_data:
                self.sgd_update(data)  # SGD优化
            # 计算RMSE并保存
            train_rmse.append(self.rmse(train_data))
            test_rmse.append(self.rmse(test_data))
            print('epoch:{}, train_rmse:{}, test_rmse:{}'.format(self.e, self.rmse(train_data), self.rmse(test_data)))
        return train_rmse, test_rmse

    def sgd_update(self, data):
        """
        :param data: 计算梯度的数据集
        """
        i, j, r_ij = data[0], data[1], data[2]
        err = float(r_ij) - np.dot(self.U[i].T, self.V[j])
        self.U[i] -= self.lr * (self.lam_u * self.U[i] - err * self.V[j])
        self.V[j] -= self.lr * (self.lam_v * self.V[j] - err * self.U[i])

    def rmse(self, data):
        """
        :param data: 数据集
        :return: RMSE
        """
        rmse = 0.0
        for i, j, r_ij in data:
            # err = 0.5 * (r_ij - np.dot(self.U[i].T, self.V[j])) ** 2 + 0.5 * self.lam_u * np.linalg.norm(
            #     self.U[i]) + 0.5 * self.lam_v * np.linalg.norm(self.V[j])
            rmse += (r_ij - np.dot(self.U[i].T, self.V[j])) ** 2
        rmse = math.sqrt(rmse / len(data))
        return rmse

    def top_k(self, uid, k):
        """
        :param uid: 需要计算的用户id
        :param k: 取多少个电影候选
        :return: k个最佳电影id
        """
        if (uid < 0) or (uid > self.n_users):
            print("用户id错误")
            return 0
        ratings = np.zeros(self.n_movies)
        r_k = np.zeros(k)
        idx_k = np.zeros(k)
        top_k = np.zeros((k, 2))
        for mid in range(self.n_movies):
            ratings[mid] = np.dot(self.U[uid], self.V[mid])
        top_k[:, 0] = np.argsort(ratings)[::-1][0:k]
        top_k[:, 1] = np.sort(ratings)[::-1][0:k]
        return top_k
