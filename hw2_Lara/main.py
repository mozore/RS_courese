import model
import data_loader
import torch

train_loader, neg_loader, neg_len = data_loader.load_train_data()
g = model.Generator()
d = model.Discriminator()
g_optim = torch.optim.Adam(g.parameters(), lr=0.001, weight_decay=0)
d_optim = torch.optim.Adam(d.parameters(), lr=0.001, weight_decay=0)
model.train(g, d, train_loader, neg_loader, 100, g_optim, d_optim, neg_len)
