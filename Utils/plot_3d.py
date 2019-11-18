# -*- coding: utf-8 -*-
# @Author  : LG

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def plot_3d(xyz, classes=None, s=0.5):
    assert isinstance(xyz, torch.Tensor) and (xyz.dim()==2)
    if classes is not None:
        assert isinstance(classes, torch.Tensor) and (classes.dim()==1)
        fig = plt.figure()
        ax = fig.gca(projection='3d')  # get current axes
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=s, c=classes)
        plt.show()
        return True
    else:
        fig = plt.figure()
        ax = fig.gca(projection='3d')  # get current axes
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=s)
        plt.show()
        return True

