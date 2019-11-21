# -*- coding: utf-8 -*-
# @Author  : LG

from torch import nn
import torch
from Model.Structs.Model_Utils import *
from torch.nn import functional as F

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        """

        :param npoint:      全局采样点
        :param radius_list: 局部采样半径
        :param nsample_list:局部采样点数
        :param in_channel:  输入通道数
        :param mlp_list:    输出通道数列表
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))   # 特征提取使用1x1卷积实现. 可以理解为全连接.但是可以省却view操作.
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz:torch.Tensor, points:torch.Tensor =None):
        """

        :param xyz:         [B, 3 ,N]
        :param features:    [B, D, N]
        :return:
        """
        xyz = xyz.transpose(1, 2)   # [B, N, C]
        if points is not None:
            points = points.transpose(1,2)  # [B, N, D]

        B, N, C =xyz.size()
        S = self.npoint
        new_xyz = index_points(xyz,furthest_point_sampling(xyz, S))     # [B, S, C] 提取全局采样点
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)   # [B, S, K]
            grouped_xyz = index_points(xyz, group_idx)              # [B, S, K, C]  局部采样点
            grouped_xyz -= new_xyz.view(B, S, 1, C)                 # 这里可以理解为数据归一化
            if points is not None:
                grouped_points = index_points(points, group_idx)    # [B, N, k, D]  如有特征点,则对特征点进行对应的采样
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)   # [B, S, K, C+D]
            else:
                grouped_points = grouped_xyz    # [B, S, K, C]

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C, K, S] or [B, C+D, K, S]  K=nsample, S=npoint
            for j in range(len(self.conv_blocks[i])):   # 进行特征提取
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))   # [B, mlp_list[-1], K, S] or [B, mlp_list[-1], K, S]
            new_points = torch.max(grouped_points, 2)[0]    # [B, mlp_list[-1], S] 这里在第三个维度上取最大的. 这里可以理解为最大池化
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)   # [B, sum(mlp_list[-1]), S]
        return new_xyz, new_points_concat   # 在输出时, 需将提取的特征 以及对应的采样点xyz返回

if __name__ == '__main__':
    sa = PointNetSetAbstractionMsg(npoint=1024,
                                   radius_list=[0.05, 0.1],
                                   nsample_list=[16, 32],
                                   in_channel=3,
                                   mlp_list=[[16, 16, 32], [32, 32, 64]])
    print(sa)
    x = torch.randn(size=(1, 3, 10000))
    new_xyz, new_points_concat = sa(x)

    print(new_xyz.size())
    print(new_points_concat.size())
