# -- coding: utf-8 --
import numpy as np
import load_data
from pmf import PMF
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_path = 'data/u.data'
    data, n_users, n_movies = load_data.load_data(file_path)
    train_data, test_data = load_data.train_test_split(data)
    n_epoches = 100
    pmf = PMF(n_feat=20, lr=0.005, lam_u=0.1, lam_v=0.1, n_epoches=n_epoches)
    train_rmse, test_rmse = pmf.fit(train_data, test_data, n_users, n_movies)
    # 画RMSE曲线图
    plt.plot(range(n_epoches), train_rmse, marker='o', label='Training Data')
    plt.plot(range(n_epoches), test_rmse, marker='v', label='Test Data')
    plt.title('PMF in movielens RMSE curve')
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()
