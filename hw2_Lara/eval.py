import torch
import pandas as pd
import numpy as np
import model
import data_loader


def get_similar_user(fake_user, user_emb, k):
    A = np.matmul(fake_user, user_emb.T)
    index = np.argsort(-A)
    return index[:, 0:k]


def dcg_at_k(r, k):
    r = np.asfarray(r)

    val = np.sum((np.power(2, r) - 1) / (np.log2(np.arange(2, r.size + 2))))
    return val


# NDCG
def ndcg_at_k(test_item, fake_user, user_emb, ui_matrix, k):
    similar_user = get_similar_user(fake_user, user_emb, k)
    sum = 0.0
    for test_mid, k_user in zip(test_item, similar_user):
        r = []
        for uid in k_user:
            r.append(ui_matrix[uid][test_mid])
        # 理想排序
        r_ideal = sorted(r, reverse=True)
        dcg_ideal = dcg_at_k(r_ideal, k)
        if dcg_ideal != 0:
            sum += (dcg_at_k(r, k) / dcg_ideal)
    return sum / test_item.__len__()


# precision
def p_at_k(test_item, fake_user, user_emb, ui_matrix, k):
    similar_user = get_similar_user(fake_user, user_emb, k)
    count = 0
    for test_mid, k_user in zip(test_item, similar_user):
        for uid in k_user:
            if ui_matrix[uid, test_mid] == 1:
                count += 1
    return count / (test_item.__len__() * k)


# 计算map
def map_at_k(test_item, fake_user, user_emb, ui_matrix, k):
    similar_user = get_similar_user(fake_user, user_emb, k)
    sum = 0.0
    for test_mid, k_user in zip(test_item, similar_user):
        count = 0
        tmp = 0.0
        r = []
        for uid in k_user:
            r.append(ui_matrix[uid][test_mid])
        for i in range(len(r)):
            if r[i] == 1:
                count += 1
                tmp += count / (i + 1)
        if count != 0:
            sum += tmp / count
    return sum / (test_item.__len__())


# 测试
def test(fake_user):
    user_emb, ui_matrix = data_loader.load_user_info()
    test_item, test_attribute = data_loader.load_test_data()
    # similar_uid = get_similar_user(fake_user, user_emb, 20)
    ndcg_10 = ndcg_at_k(test_item, fake_user, user_emb, ui_matrix, 10)
    ndcg_20 = ndcg_at_k(test_item, fake_user, user_emb, ui_matrix, 20)
    p_10 = p_at_k(test_item, fake_user, user_emb, ui_matrix, 10)
    p_20 = p_at_k(test_item, fake_user, user_emb, ui_matrix, 20)
    map_10 = map_at_k(test_item, fake_user, user_emb, ui_matrix, 10)
    map_20 = map_at_k(test_item, fake_user, user_emb, ui_matrix, 20)
    print('test:ndcg@10:{:.3f}, ndcg@20:{:.3f}, p@10:{:.3f}, p@20:{:.3f}, map@10:{:.3f}, map@20:{:.3f}'
          .format(ndcg_10, ndcg_20, p_10, p_20, map_10, map_20))
