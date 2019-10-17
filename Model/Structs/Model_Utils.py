# -*- coding: utf-8 -*-
# @Author  : LG

import torch
__all__ = ['index_points', 'query_ball_point', 'square_distance', 'furthest_point_sampling']

def furthest_point_sampling(xyz:torch.Tensor, nsamples:int) -> torch.Tensor:
    """
    试图从xyz 中取出 nsamples 个点,且点之间的距离要尽可能大, 返回点在点云中的索引
    :param xyz:     # 点云数据, size:[B, N, 3]
    :param nsamples:    # 要选取的点数量, int
    :return:    选取点在xyz中的索引, size:[B, nsamples]
    """
    B, N, C = xyz.size()
    device = xyz.device
    idx = torch.zeros((B,nsamples),device=device, dtype=torch.long)   # 用于存储最后选取点的索引.
    tmp = torch.ones((B,N),device=device).fill_(1e10)   # 距离矩阵,以1e10作为初始化.
    farthest = torch.randint(high=N, size = (B,)).to(device) # 用随机初始化的最远点
    batch_indices = torch.arange(B).to(device)
    for i in range(nsamples):
        # 依次更新idx中的数据.共nsamples个
        idx[:,i] = farthest
        # 取出第i个点的xyz坐标
        centroids = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 计算距离
        dist = torch.sum((xyz-centroids)**2, -1)
        tmp = tmp.type_as(dist)
        mask = dist < tmp
        # 更新距离矩阵中数据,矩阵中距离在慢慢变小
        tmp[mask] = dist[mask]

        # 更新最远点
        farthest = torch.max(tmp,-1)[1]
    return idx


def index_points(xyz:torch.Tensor, idx:torch.Tensor) -> torch.Tensor:
    """
    按照索引idx 从xyz中取对应的点.
    :param xyz: 点云数据    [B, N, C]
    :param idx: 要选取的点的索引    [B, S]
    :return:    返回选取的点  [B, S, C]
    """
    device = xyz.device
    B = xyz.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = xyz[batch_indices, idx, :]
    return new_points


def square_distance(src:torch.Tensor, dst:torch.Tensor) -> torch.Tensor:
    """
    计算俩组坐标之间的欧氏距离,  (x-y)**2 = x**2 + y**2 - 2*xy
    :param src: [B, N, 3]
    :param dst: [B, S, 3]
    :return:    俩组坐标之间的俩俩对应距离的矩阵 size: [B, N, S]
    """
    B, N, C = src.shape
    S = dst.size(1)
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # 2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1) # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, S) # xm*xm + ym*ym + zm*zm
    return dist


def query_ball_point(radius:float, nsample:int, xyz:torch.Tensor, new_xyz:torch.Tensor) -> torch.Tensor:
    B, N, C = xyz.size()
    S = new_xyz.size(1)
    device = xyz.device
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat([B, S, 1])
    # xyz 与 xyz_new 之间坐标俩俩对应的距离矩阵 [B, N, S]
    sqrdists = square_distance(new_xyz, xyz)
    # 大于radius**2的,将group_idx 之间置为N.
    group_idx[sqrdists > radius**2] = N
    # 做升序排列，取出前nsample个点
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # 对于数据不足的情况,直接将等于N的点替换为第一个点的值
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

