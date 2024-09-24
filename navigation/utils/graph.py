import numpy as np
import torch
import torch.nn as nn 
import cv2
import quaternion
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
import matplotlib.cm as cm 
from utils.model import get_grid, ChannelPool
from torch.nn import functional as F


class Node(object):
    def __init__(self, idname, rgb, depth, pose, proj_map, exp_map, img_feat):
        self.is_start = False 
        self.id = idname
        self.rgb = rgb
        self.depth = depth
        self.pose = pose
        self.proj_map = proj_map
        self.exp_map = exp_map
        self.img_feat = img_feat

    def set_to_start(self):
        self.is_start = True

class Edge(object):
    def __init__(self, node1, node2, sensor_pose, gt_sensor_pose, edge_type):
        self.nodes = (node1, node2)
        self.ids = (node1.id, node2.id)
        self.sensor_pose = sensor_pose
        self.gt_sensor_pose = gt_sensor_pose
        self.edge_type = edge_type  # 0 if traj_edge # 1 if landmark_edge



class GraphMap(object):
    def __init__(self, device):

        self.nodes = set()
        self.node_by_id = {}
        self.edges = set()
        self.edge_by_id = {}
        self.rep_nodes = set() 

        self.device = device
        self.img_feat_size = 512 
        # self.resnet = load_places_resnet(device)

    def get_rep_node_feat(self):        
        rep_feats = np.zeros((len(self.rep_nodes), 1, self.img_feat_size))
        for n in range(len(self.rep_nodes)):
            rep_node_id = list(self.rep_nodes)[n]
            rep_feats[n] = self.node_by_id[str(rep_node_id)].img_feat
        return rep_feats 


    def node_at(self, node_id): 
        return self.node_by_id[str(node_id)]
    
    def edge_at(self, id0, id1): 
        return self.edge_by_id[str(id0), str(id1)]


    def add_rep_node(self, node_id):
        self.rep_nodes.add(node_id)


    def add_features(self, feats):
        self.features = feats

    def add_robot_node(self, rgb, depth, pose, proj_map, exp_map, img_feat):

        node_id = str(len(self.nodes))    
        node = Node(node_id, rgb, depth, pose, proj_map, exp_map, img_feat)
        self.nodes.add(node)

        if node_id == '0':
            node.set_to_start()
        self.node_by_id[node_id] = node 
        return node_id

    def add_robot_edge(self, node1, node2, sensor_pose, gt_sensor_pose, edge_type):
        edge = Edge(node1, node2, sensor_pose, gt_sensor_pose, edge_type)
        self.edges.add(edge)
        self.edge_by_id[node1.id, node2.id] = edge

 


class Graph_Mapper(nn.Module):

    def __init__(self, args):
        super(Graph_Mapper, self).__init__()

        self.device = args.device
        self.resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm 
        self.vision_range = args.vision_range

        self.pool = ChannelPool(1)

        num_processes = 1
        self.agent_view = torch.zeros(num_processes, 2,
                                      self.map_size_cm // self.resolution,
                                      self.map_size_cm // self.resolution
                                      ).float().to(self.device)

        self.num_scenes = num_processes



    def forward(self, gt_proj, gt_exp, poses, maps, explored, current_poses, b_selected_f, freq_map, sim_score=0):


        pred = torch.zeros((self.num_scenes,2,self.vision_range,self.vision_range))
        for i in range(self.num_scenes):
            pred[i] = torch.stack((gt_proj[i], gt_exp[i]))


        agent_view = self.agent_view.detach_()
        agent_view.fill_(0.)

        x1 = self.map_size_cm // (self.resolution * 2) \
                - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, :, y1:y2, x1:x2] = pred

        def get_new_pose_batch(pose, rel_pose_change):
            pose[:, 1] += rel_pose_change[:, 0] * \
                            torch.sin(pose[:, 2] / 57.29577951308232) \
                            + rel_pose_change[:, 1] * \
                            torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                            torch.cos(pose[:, 2] / 57.29577951308232) \
                            - rel_pose_change[:, 1] * \
                            torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(current_poses,
                                            poses)

        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution
                            - self.map_size_cm \
                            // (self.resolution * 2)) \
                            / (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                        self.device)

        rotated = F.grid_sample(agent_view, rot_mat)
        translated = F.grid_sample(rotated, trans_mat)

        if b_selected_f == 0 : 
            translated = translated*0
        maps2 = torch.cat((maps.unsqueeze(1),
                            translated[:, :1, :, :]), 1)
        explored2 = torch.cat((explored.unsqueeze(1),
                                translated[:, 1:, :, :]), 1)

        
        map_pred = self.pool(maps2).squeeze(1)
        exp_pred = self.pool(explored2).squeeze(1)

        if sim_score == 0: 
            freq_map[:,:2] += translated.detach().clone()
        else: 
            scored_translated = (translated.detach().clone() > 0.5)*sim_score
            freq_map[:,:2] += scored_translated
            # print("ehrre")

        
        return map_pred, exp_pred, current_poses, freq_map
