# -*- coding: utf-8 -*-
# @Author  : LG

from Data.Dataset import indoor3d_Dataset
from torch.utils.data import DataLoader
from Model.Pointnet2 import PointnetMSG
from torch import nn
import torch
from torch.nn import DataParallel
from Utils.visdom_op import setup_visdom, visdom_line

EPOCH = 100
LR = 0.0001
vis = setup_visdom()

train_data = indoor3d_Dataset(is_train=True, data_root='Data', test_area=5)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

model = PointnetMSG(xyz_channel=3, data_channel=6, num_classes=13)
model.load_state_dict(torch.load('/home/super/PycharmProjects/pointnet2/Weights/model_40000.pkl'))
model = model.to('cuda')
model = DataParallel(model, device_ids=[0, 1])

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

num = 0
for epoch in range(EPOCH):
    for step, (data, label) in enumerate(train_loader):
        print(data.size())
        print(label.size())
        data = data.transpose(1, 2)
        label = label.view(-1).long().to('cuda')

        pred = model(xyz=data[:, :3, :], points=data[:, 3:, :])
        pred = pred.view(-1, pred.size(-1))

        loss = loss_fn(pred, label)
        num += 1
        visdom_line(vis, y=[loss], x=num,win_name='pointnet')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch : {} | step : {} | loss : {:.4f} '.format(epoch, step, loss.cpu().item()))
        if num%2000 == 0:
            torch.save(model.module.state_dict(), '{}/model_{}.pkl'.format('Weights', num))
