# -*- coding: utf-8 -*-
# @Author  : LG

import torch
from torch import nn
from Model.Structs.Model_Utils import *
from torch.nn import functional as F


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        B, N, C = xyz1.size()
        _, S, _ = xyz2.size()

        if S == 1:
            interpolated_points = points2.repeat((1, N, 1))
        else:
            dists = square_distance(xyz1, xyz2)
            dicts, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dists[dists<1e-10] = 1e-10
            weight = dists.reciprocal()
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)

        for i ,conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


if __name__ == '__main__':

    x = torch.randn(size=(1, 3, 10000))
    points = torch.randn(size=(1, 6, 10000))

    new_x = torch.randn(size=(1, 3, 1024))
    new_points = torch.randn(size=(1, 96, 1024))

    fp = PointNetFeaturePropagation(in_channel=96+6, mlp=[32, 64])
    print(fp)
    new_points = fp(x, new_x, points, new_points)
    print(new_points.size())
