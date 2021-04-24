from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import pandas as pd
import numpy as np


class LaraDataset(Dataset):
    """
    自定义的Dataset
    """

    def __init__(self, train_csv, user_emb_csv):
        # 读取数据 (uid, mid, attr)
        self.train_data = pd.read_csv(train_csv, header=None)
        self.uid = self.train_data.loc[:, 0]
        self.mid = self.train_data.loc[:, 1]
        self.attr = self.train_data.loc[:, 2]
        # 读取用户embedding
        self.user_emb_data = pd.read_csv(user_emb_csv, header=None)
        self.user_emb_data = np.array(self.user_emb_data)

    def __getitem__(self, idx):
        uid = self.uid[idx]
        mid = self.mid[idx]
        attr = self.attr[idx][1: -1].split()
        attr = torch.tensor(list(map(int, attr)), dtype=torch.long)
        # print(attr)
        user_emb = self.user_emb_data[uid]
        return uid, mid, attr, user_emb

    def __len__(self):
        return len(self.train_data)


# a = LaraDataset('data/train/train_data.csv', 'data/train/user_emb.csv')
# a.__getitem__(0)
#
#
# class LaraDataLoader

# 加载训练数据和负例数据
def load_train_data():
    train_dataset = LaraDataset('data/train/train_data.csv', 'data/train/user_emb.csv')
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    neg_dataset = LaraDataset('data/train/neg_data.csv', 'data/train/user_emb.csv')
    neg_loader = DataLoader(neg_dataset, batch_size=1024, shuffle=True, num_workers=0)
    return train_loader, neg_loader, neg_dataset.__len__()


# 加载测试数据
def load_test_data():
    # 测试物品
    test_item = pd.read_csv('data/test/test_item.csv', header=None)
    test_item = np.array(test_item)
    # 测试物品的属性
    test_attribute = pd.read_csv('data/test/test_attribute.csv', header=None)
    test_attribute = np.array(test_attribute)
    return test_item, test_attribute


def load_user_info():
    # 用户嵌入，（train test不用一个就离谱）
    user_emb = pd.read_csv('data/test/user_attribute.csv', header=None)
    user_emb = np.array(user_emb)
    ui_matrix = pd.read_csv('data/test/ui_matrix.csv', header=None)
    ui_matrix = np.array(ui_matrix)
    return user_emb, ui_matrix
