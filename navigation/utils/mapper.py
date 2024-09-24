import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np

from utils.model import get_grid, ChannelPool
import matplotlib.pyplot as plt 



class Remap_Module(nn.Module):
    """
    """

    def __init__(self, args):
        super(Remap_Module, self).__init__()

        self.device = args.device
        self.resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm 
        self.vision_range = args.vision_range

        self.pool = ChannelPool(1)
        self.agent_view = torch.zeros(1, 2,
                                      self.map_size_cm // self.resolution,
                                      self.map_size_cm // self.resolution
                                      ).float().to(self.device)

    def forward(self, fp_proj, fp_explored,  current_poses, maps, explored): 


        pred = torch.stack((fp_proj, fp_explored), axis = 1)

        agent_view = self.agent_view.detach_()
        agent_view.fill_(0.)

        x1 = self.map_size_cm // (self.resolution * 2)  - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, :, y1:y2, x1:x2] = pred

        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution - self.map_size_cm \
                                // (self.resolution * 2)) \
                                / (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)

        rotated = F.grid_sample(agent_view, rot_mat)
        translated = F.grid_sample(rotated, trans_mat)

        maps2 = torch.cat((maps.unsqueeze(1),  translated[:, :1, :, :]), 1)
        explored2 = torch.cat((explored.unsqueeze(1), translated[:, 1:, :, :]), 1)

        map_pred = self.pool(maps2).squeeze(1)
        exp_pred = self.pool(explored2).squeeze(1)

        return map_pred, exp_pred

