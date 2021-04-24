import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import eval
import data_loader

# 超参数设定
alpha = 0
attr_num = 18  # 属性数量，movie 18个类别
attr_dim = 5  # 属性嵌入维数
batch_size = 1024  # batch大小
hidden_dim = 100  # 隐藏层维度
user_emb_dim = attr_num  # 训练的表示
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_attr_matrix = nn.Embedding(2 * attr_num, attr_dim)
        # 三层非线性
        self.l1 = nn.Linear(attr_num * attr_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim)
        self.tanh = nn.Tanh()
        self.init_params()

    def init_params(self):
        # 参数初始化
        for m in self.G_attr_matrix.modules():
            nn.init.xavier_normal_(m.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.xavier_normal_(m.bias.unsqueeze(0))

    def forward(self, attr_id):
        # 挑出18个类别表示
        attr_present = self.G_attr_matrix(attr_id)
        # attr_feat = torch.reshape(attr_present, [-1, attr_num * attr_dim])
        attr_feat = attr_present.view(-1, attr_num * attr_dim)
        # 三次非线性变化后得到用户的生成表示
        o1 = self.tanh(self.l1(attr_feat))
        o2 = self.tanh(self.l2(o1))
        fake_user = self.tanh(self.l3(o2))
        return fake_user


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_attr_matrix = nn.Embedding(attr_num * 2, attr_dim)
        # 三次非线性
        self.l1 = nn.Linear(attr_num * attr_dim + user_emb_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_params()

    def init_params(self):
        for m in self.D_attr_matrix.modules():
            nn.init.xavier_normal_(m.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.xavier_normal_(m.bias.unsqueeze(0))

    def forward(self, attr_id, user_emb):
        attr_id = attr_id.long()  # 转为需要的long类型
        attr_present = self.D_attr_matrix(attr_id)
        # attr_feat = torch.reshape(attr_present, [-1, attr_num * attr_dim])
        attr_feat = attr_present.view(-1, attr_num * attr_dim)
        emb = torch.cat((attr_feat, user_emb), 1)  # 拼接用户表示和属性表示
        emb = emb.float()  # scalar type 转为float
        # 非线性变化得到判别概率
        o1 = self.tanh(self.l1(emb))
        o2 = self.tanh(self.l2(o1))
        d_logit = self.l3(o2)
        d_prob = self.sigmoid(d_logit)
        return d_prob, d_logit


# 训练数据
def train(g, d, train_loader, neg_loader, epoches, g_optim, d_optim, neg_lens):
    g = g.to(device)
    d = d.to(device)
    time.sleep(0.1)
    print("start training on {}".format(device))
    time.sleep(0.1)
    bce_loss = torch.nn.BCELoss()
    # 训练判别器D
    for e in tqdm(range(epoches)):
        start_time = time.time()
        idx = 0
        d_loss = 0.0
        neg_iter = neg_loader.__iter__()
        # 训练判别器d
        for _, _, real_attr, real_user_emb in train_loader:
            if idx > neg_lens:
                break
            _, _, neg_attr, neg_user_emb = neg_iter.next()
            # 正例的属性和用户嵌入
            real_attr = real_attr.to(device)
            real_user_emb = real_user_emb.to(device)
            # 负例的属性和用户嵌入
            neg_attr = neg_attr.to(device)
            neg_user_emb = neg_user_emb.to(device)
            # 生成器生成虚拟用户嵌入
            fake_user_emb = g(real_attr)
            fake_user_emb = fake_user_emb.to(device)
            # 判别器判别
            d_real, d_logit_real = d(real_attr, real_user_emb)
            d_fake, d_logit_fake = d(real_attr, fake_user_emb)
            d_neg, d_logit_neg = d(neg_attr, neg_user_emb)
            # 计算d_loss
            d_optim.zero_grad()
            d_loss_real = bce_loss(d_real, torch.ones_like(d_real))
            d_loss_fake = bce_loss(d_fake, torch.zeros_like(d_fake))
            d_loss_neg = bce_loss(d_neg, torch.zeros_like(d_neg))
            d_loss = torch.mean(d_loss_real + d_loss_fake + d_loss_neg)
            d_loss.backward()
            d_optim.step()
            idx += batch_size
        # 训练生成器g
        g_loss = 0.0
        for uid, mid, attr, user_emb in train_loader:
            g_optim.zero_grad()
            attr = attr.to(device)
            # 生成虚拟用户嵌入
            fake_user_emb = g(attr)
            fake_user_emb = fake_user_emb.to(device)
            # 算loss
            d_fake, d_logit_fake = d(attr, fake_user_emb)
            g_loss = bce_loss(d_fake, torch.ones_like(d_fake))
            g_loss.backward()
            g_optim.step()
        end_time = time.time()
        print(
            "\nepoch:{}: time:{:.2f}, d_loss:{:.3f}, g_loss:{:.3f}".format(e + 1, end_time - start_time, d_loss,
                                                                           g_loss))
        # test
        test_item, test_attribute = data_loader.load_test_data()
        test_item = torch.tensor(test_item).to(device)
        test_attribute = torch.tensor(test_attribute, dtype=torch.long).to(device)
        fake_user = g(test_attribute)
        eval.test(fake_user.cpu().detach().numpy())
        time.sleep(0.1)
