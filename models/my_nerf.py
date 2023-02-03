import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time

class CheatNeRF():
    def __init__(self, nerf):
        super(CheatNeRF, self).__init__()
        self.nerf = nerf
    
    def query(self, pts_xyz):
        return self.nerf(pts_xyz, torch.zeros_like(pts_xyz))

class MyNeRF():
    def __init__(self):
        RS = 128
        super(MyNeRF, self).__init__()
        self.volume_sigma = torch.zeros((RS, RS, RS))
        self.volume_color = torch.zeros((RS, RS, RS, 3))
        self.octree = None

    def save(self, pts_xyz, sigma, color):
        RS = 128

        X_index = ((pts_xyz[:, 0] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
        Y_index = ((pts_xyz[:, 1] - 0.75) * 4 * RS).clamp(0, RS - 1).long()
        Z_index = ((pts_xyz[:, 2] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
        
        self.volume_sigma[X_index, Y_index, Z_index] = sigma[:, 0]
        self.volume_color[X_index, Y_index, Z_index] = color[:, :]

        the_object = {
            "voxel_size": RS,
            "volume_sigma": self.volume_sigma,
            "volume_color": self.volume_color,
        }
        torch.save(the_object, "data_" + str(RS) + ".pth")

    def query(self, pts_xyz):
        RS = 128

        N, _ = pts_xyz.shape
        sigma = torch.zeros(N, 1, device=pts_xyz.device)
        color = torch.zeros(N, 3, device=pts_xyz.device)

        if self.octree is not None:
            color, sigma = torch.split(self.octree[pts_xyz], 3, dim=1)

        else:
            X_index = ((pts_xyz[:, 0] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
            Y_index = ((pts_xyz[:, 1] - 0.75) * 4 * RS).clamp(0, RS - 1).long()
            Z_index = ((pts_xyz[:, 2] + 0.125) * 4 * RS).clamp(0, RS - 1).long()

            sigma[:, 0] = self.volume_sigma[X_index, Y_index, Z_index]
            color[:, :] = self.volume_color[X_index, Y_index, Z_index]

        return sigma, color