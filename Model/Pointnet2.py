# -*- coding: utf-8 -*-
# @Author  : LG

import torch
from torch import nn
from Model.Structs import PointNetSetAbstractionMsg, PointNetFeaturePropagation
from torch.nn import functional as F

class PointnetMSG(nn.Module):
    def __init__(self, xyz_channel=3, data_channel=0, num_classes=13):
        super(PointnetMSG, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(npoint=1024,
                                             radius_list=[0.1,0.2],
                                             nsample_list=[16,32],
                                             in_channel=xyz_channel + data_channel,
                                             mlp_list=[[32,32,64],[32,48,64]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256,
                                             radius_list=[0.2, 0.4],
                                             nsample_list=[16, 32],
                                             in_channel=128+xyz_channel,
                                             mlp_list=[[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(npoint=64,
                                             radius_list=[0.4, 0.8],
                                             nsample_list=[16, 32],
                                             in_channel=256+xyz_channel,
                                             mlp_list=[[128, 128, 256], [128, 192, 256]])

        self.fp3 = PointNetFeaturePropagation(in_channel=768, mlp=[512, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+data_channel, mlp=[128, 128])

        self.conv1 = nn.Conv1d(128,128,1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz:torch.Tensor, points:torch.Tensor = None):
        l1_xyz ,l1_points = self.sa1(xyz, points)
        # print('l1_xyz', l1_xyz.size(), l1_points.size())
        l2_xyz, l2_points = self.sa2(l1_xyz ,l1_points)
        # print('l2_xyz', l2_xyz.size(), l2_points.size())
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print('l3_xyz', l3_xyz.size() ,l3_points.size())
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print('l2_points', l2_points.size())
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print('l1_points', l1_points.size())
        l0_points = self.fp1(xyz, l1_xyz, points, l1_points)
        # print('l0_points', l0_points.size())

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        # print(x.size())
        return x
if __name__ == '__main__':
    from torch.nn import DataParallel

    pointnet = PointnetMSG(xyz_channel=3, data_channel=5).to('cuda:0')

    x1 = torch.randn(size=(2, 3, 10000)).to('cuda:0')
    x2 = torch.randn(size=(2, 5, 10000)).to('cuda:0')

    print(pointnet)
    y = pointnet(x1,x2)
