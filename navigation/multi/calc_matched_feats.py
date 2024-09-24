import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

from utils.model import get_grid, ChannelPool
import matplotlib.pyplot as plt
import numpy as np 
import env.utils.depth_utils as du
from scipy.spatial.transform import Rotation

from skimage.draw import line

import matplotlib
# matplotlib.use('TkAgg')

color_map = {'red': [1,0,0],
                    'blue': [0,0,1] ,
                    'green': [0,1,0],
                    'yellow': [1,1,0],
                    'pink': [1,0,1],
                    'white': [1,1,1]}

def line_interval(number):
    no_of_interval = 10
    interval = number / no_of_interval

    values = []
    for i in range(no_of_interval + 1):
        value = interval * i
        values.append(value)

    values = values[1:-2]

    return values

def normalize_angle(angle_radians):
    angle_degrees = np.rad2deg(angle_radians)
    normalized_angle = angle_degrees % 360
    return normalized_angle


class CALC_FEAT_MATCH(nn.Module):

    def __init__(self, args):
        super(CALC_FEAT_MATCH, self).__init__()

        params = {}

        params['agent_view_angle'] = 0 
        # params['agent_min_z'] = 25
        # params['agent_max_z'] = 150
        # params['obs_threshold'] = 1.0
        # params['map_size'] = 2400
        # map_size = params['map_size']

        # import pdb; pdb.set_trace()
        fov = args.hfov
        self.device = args.device
        self.agent_height = args.camera_height * 100 
        self.agent_view_angle = params['agent_view_angle']
        # self.vision_range = params['vision_range']
        self.vision_range = 320
        self.frame_width = args.env_frame_width
        self.frame_height = args.env_frame_height
        # self.du_scale = params['du_scale']
        # self.resolution = params['resolution']
        self.du_scale = 1
        self.resolution = 1
        # self.z_bins = [params['agent_min_z'], params['agent_max_z']]
        self.z_bins = [50,180]
        self.obs_threshold = args.obs_threshold
        self.camera_matrix = du.get_camera_matrix(
            self.frame_width,
            self.frame_height,
            fov) 
        
        map_size = args.map_size_cm
        num_processes = 1
        self.map_size = map_size
        self.map = torch.zeros((num_processes, 3 ,map_size // self.resolution,  map_size // self.resolution)).to(self.device)
        self.tf_map = torch.zeros((map_size // self.resolution, map_size // self.resolution, 3)).to(self.device)
        self.check_map = torch.zeros((map_size // self.resolution, map_size // self.resolution, 3)).to(self.device)

        self.est_map = np.zeros((map_size // self.resolution,
                             map_size // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)
        
        self.nocs_map = np.zeros((map_size // self.resolution,
                             map_size // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)


        self.agent_view = torch.zeros(num_processes, 2,
                                      self.map_size // self.resolution,
                                      self.map_size // self.resolution
                                      ).float().to(self.device)
        self.pool = ChannelPool(1)
        self.rotation_2d0 = 0 
        self.rotation_2d1 = 0
        self.angle_deg_to_centroid0 = 0 
        self.c0top0 = 0 
        self.angle_deg_to_centroid1 = 0 
        self.c1top1 = 0 
        self.centroid_x0 = 0 
        self.centroid_y0 = 0 

        self.angle0 = 0 
        self.angle1 = 0 
        self.dist0 = 0 
        self.dist1 = 0 
        self.landmark_list = []
        self.landmarkno = 0 
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]

    def masked_area_depth(self, depth, pred_mask):
        depth = depth[:, :, 0]*1
        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask3 = pred_mask[0].cpu().numpy()
        depth[mask3 == False] = 0 

        for i in range(depth.shape[1]):
            depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

        mask1 = depth == 0
        depth[mask1] = np.NaN
        depth = depth*1000.
        return depth 



    def keypoints_depth(self, depth, kpts):
        kpt_depth = np.zeros(depth.shape)
        for kpt in kpts: 
            y, x = kpt
            kpt_depth[x,y,:] = depth[x,y,:]

        depth = kpt_depth[:, :, 0]*1
        mask2 = depth > 0.99
        depth[mask2] = 0.

        for i in range(depth.shape[1]):
            depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

        mask1 = depth == 0
        depth[mask1] = np.NaN
        depth = depth*1000.
        return depth



    def keypoint_depth(self, depth, kpt):

        y, x = kpt
        kpt_depth = np.zeros(depth.shape)
        kpt_depth[x,y,:] = depth[x,y,:]

        depth = kpt_depth[:, :, 0]*1
        mask2 = depth > 0.99
        depth[mask2] = 0.

        for i in range(depth.shape[1]):
            depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

        mask1 = depth == 0
        depth[mask1] = np.NaN
        depth = depth*1000.
        return depth

    def preprocess_depth(self, depth):
        depth = depth[:, :, 0]*1
        mask2 = depth > 0.99
        depth[mask2] = 0.

        for i in range(depth.shape[1]):
            depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

        mask1 = depth == 0
        depth[mask1] = np.NaN
        depth = depth*1000.
        return depth
        
    def ego_mapper(self, depth): 
        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix, scale=self.du_scale)
        agent_view = du.transform_camera_view(point_cloud, self.agent_height, self.agent_view_angle)
        agent_view_centered = du.transform_pose(agent_view, self.shift_loc)
        agent_view_flat = du.bin_points(
            agent_view_centered,
            self.vision_range,
            self.z_bins,
            self.resolution)
        agent_view_cropped = agent_view_flat[:, :, 1]



        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0


        return agent_view, agent_view_cropped, agent_view_explored



    def get_new_pose(self, current_pose, distance, angle_degrees):
        angle_radians = np.deg2rad(angle_degrees)
        # Calculate the new position
        delta_x = distance * np.cos(angle_radians)
        delta_y = distance * np.sin(angle_radians)
        rotation_matrix = np.array([[np.cos(np.deg2rad(current_pose[2])), -np.sin(np.deg2rad(current_pose[2]))],
                                        [np.sin(np.deg2rad(current_pose[2])), np.cos(np.deg2rad(current_pose[2]))]])
        delta = np.dot(rotation_matrix, np.array([delta_x, delta_y]))
        new_position = np.array([current_pose[0], current_pose[1]]) + delta
        return new_position

        p0, p1 = [int(current_pose0[0] * 100. / self.resolution), int(current_pose0[1]* 100 / self.resolution)]
        loc_l, loc_r = max( 0, int(p1) -10 ), min( int(p1)+10, self.map.shape[0])
        loc_t, loc_d = min( int(p0)+10, self.map.shape[0] ), max( 0, int(p0)-10 )

    def map_projection(self, depth, pose):
        mapper_pose = (pose[0]*100.0, pose[1]*100.0, np.deg2rad(pose[2]))
        agent_view, proj, explored = self.ego_mapper(depth)
        geocentric_pc = du.transform_pose(agent_view, mapper_pose)
        geocentric_flat = du.bin_points(geocentric_pc,self.map.shape[0],self.z_bins,self.resolution)
        map_gt = geocentric_flat0[:, :, 1] / self.obs_threshold
        return map_gt, proj, explored

    def draw_map(self, map_1d, colorname): 
        color = color_map[colorname]
        self.map[:,:,0][map_1d >= 0.5] = color[0]
        self.map[:,:,1][map_1d >= 0.5] = color[1]
        self.map[:,:,2][map_1d >= 0.5] = color[2]

    def draw_pose(self, pose, colorname): 
        color = color_map[colorname]
        thickness = 5
        p0, p1 = [int(pose[1] * 100. / self.resolution), int(pose[0]* 100 / self.resolution)]
        loc_l, loc_r = max( 0, int(p0) - thickness), min( int(p0)+ thickness, self.map.shape[0])
        loc_t, loc_d = min( int(p1)+thickness, self.map.shape[0] ), max( 0, int(p1)-thickness )
        self.map[loc_l:loc_r, loc_d:loc_t,:] = color

    def find_new_position(self, pose):
        distance = 10
        x0 = pose[0]
        y0 = pose[1]
        theta = pose[2]
        # Convert theta to radians
        theta_rad = np.deg2rad(90 - theta)
        # Calculate the new position coordinates
        x1 = x0 + distance * np.cos(theta_rad)
        y1 = y0 + distance * np.sin(theta_rad)
        new_pose = [x1, y1, theta]
        return new_pose

    def draw_a_line_at_heading_direction(self, pose, color): 
        distance_list = line_interval(7)
        for dist in distance_list: 
            theta_rad = np.deg2rad( pose[2])
            x1 = pose[0] + dist * np.cos(theta_rad)
            y1 = pose[1] + dist * np.sin(theta_rad)
            new_pose = [x1, y1, 0]
            self.draw_pose(new_pose, color)

    def draw_a_line_at_angle(self, pose, distance, angle_degrees, color): 
        distance_list = line_interval(distance)

        for dist in distance_list: 
            theta_deg = pose[2] + angle_degrees 
            theta_rad = np.deg2rad(90 - theta_deg)
            x1 = pose[0] + dist * np.cos(theta_rad)
            y1 = pose[1] + dist * np.sin(theta_rad)
            new_pose = [x1, y1, 0]
            self.draw_pose(new_pose, color)
        return new_pose

    def find_relative_distance_and_angle(self, x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        distance = np.sqrt(dx**2 + dy**2)
        angle_deg =  90 - np.rad2deg(np.arctan2(dy, dx))
        return distance, angle_deg


    def heading_line(self, current_pose, colorname):
        distance = line_interval(4)
        for dist in distance:
            tmp_x, tmp_y = self.pose_at_dist_angle(current_pose[0,0], current_pose[0,1], current_pose[0,2], dist)
            self.mark_pose(tmp_x,tmp_y,self.tf_map, colorname)

    def get_new_pose_batch(self, pose, rel_pose_change):
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

    def mark_pose(self, p0, p1, map_3c, colorname):
        color = color_map[colorname]
        loc_r, loc_c = [int(p1 * 100.0 / self.resolution), int(p0 * 100.0 / self.resolution)]
        map_3c[ loc_r - 3:loc_r + 3, loc_c - 3:loc_c + 3, : ] = torch.tensor(color)

    # def pose_at_dist_angle (self, x0, y0, th_deg, dist):
    #     x1 = y0 + dist * torch.cos(torch.deg2rad(th_deg))
    #     y1 = x0 + dist * torch.sin(torch.deg2rad(th_deg))
    #     return x1, y1
    def pose_at_dist_angle (self, x0, y0, th_deg, dist):
        x1 = x0 + dist * torch.cos(torch.deg2rad(th_deg))
        y1 = y0 + dist * torch.sin(torch.deg2rad(th_deg))
        return x1, y1
    def get_dist_angle(self, masked_proj):
        centroid_x, centroid_y = np.mean(np.where(masked_proj == 1), axis = 1)
        y = centroid_x
        x = centroid_y - (self.vision_range * self.resolution) / 2. 
        # y, x = np.min(np.where(masked_proj == 1), axis = 1)
        # y = min_x
        # x = min_y - (self.vision_range * self.resolution) / 2.
        angle_deg = torch.tensor(np.rad2deg(np.arctan(x/y)))
        dist = np.sqrt((x)**2 + (y)**2)/100.
        return dist, angle_deg

    def kpt_dist_angle2(self, depth, kpt, current_pose, kpt_color):
        y, x = kpt
        if depth[x,y,:] <= 0.99 :
            new_depth = np.zeros(depth[:,:,0].shape)
            new_depth[:,:] = np.NaN 
            new_depth[:,y] = depth[x,y,:]
            new_depth = new_depth*1000
        
        point_cloud = du.get_point_cloud_from_z(new_depth, self.camera_matrix, scale=self.du_scale)
        agent_view = du.transform_camera_view(point_cloud, self.agent_height, self.agent_view_angle)
        agent_view_centered = du.transform_pose(agent_view, self.shift_loc)
        agent_view_flat = du.bin_points(
            agent_view_centered,
            self.vision_range,
            self.z_bins,
            self.resolution)
        agent_view_cropped = agent_view_flat[:, :, 1]



        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0





    def kpt_dist_angle(self, depth, kpt, current_pose, kpt_color): 
        kpt_depth = self.keypoint_depth(depth, kpt)
        _, kpt_proj, _ = self.ego_mapper(kpt_depth)

        preprocessed_depth = self.preprocess_depth(depth)
        agent_view, proj, explored = self.ego_mapper(preprocessed_depth)
        fp_proj = torch.from_numpy(proj).float().unsqueeze(0).unsqueeze(0)
        fp_explored = torch.from_numpy(explored).float().unsqueeze(0).unsqueeze(0)
        pred = torch.hstack((fp_proj, fp_explored)).to(self.device)
        agent_view = self.agent_view.detach_()
        agent_view.fill_(0.)

        x1 = self.map_size // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size // (self.resolution * 2)
        y2 = y1 + self.vision_range


        agent_view[:, :, y1:y2, x1:x2] = pred

        st_pose = current_pose.clone().detach()
        st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution
                                    - self.map_size \
                                    // (self.resolution * 2)) \
                                    / (self.map_size // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose.to(self.device), agent_view.size(), self.device)

        rotated = F.grid_sample(agent_view, rot_mat)
        translated = F.grid_sample(rotated, trans_mat)
        map_mask = translated[0,0].detach()
        color = color_map['white']
        self.tf_map[:,:,0][map_mask >= 0.5] = color[0]
        self.tf_map[:,:,1][map_mask >= 0.5] = color[1]
        self.tf_map[:,:,2][map_mask >= 0.5] = color[2]

        # mark keypoint and line from pose0 to keypoint
        keypoint_dist, keypoint_angle_deg = self.get_dist_angle(kpt_proj)
        angle_deg_to_keypoint = current_pose[0,2] - keypoint_angle_deg
        keypoint_x, keypoint_y = self.pose_at_dist_angle(current_pose[0,0], current_pose[0,1], angle_deg_to_keypoint , keypoint_dist)
        # if torch.isnan(keypoint_x): 
        #     return None, None, None, None, None
        # self.mark_pose( keypoint_x, keypoint_y, self.tf_map, kpt_color)
        # self.mark_pose( current_pose[0,0], current_pose[0,1], self.tf_map, 'green')

        # distance = line_interval(keypoint_dist)
        # for dist in distance:
        #     new_x, new_y = self.pose_at_dist_angle(current_pose[0,0], current_pose[0,1], angle_deg_to_keypoint , dist)
        #     self.mark_pose( new_x, new_y, self.tf_map, 'yellow')


        return True, keypoint_angle_deg, keypoint_dist, keypoint_x, keypoint_y



    def centroid_dist_angle(self, depth, kpts, current_pose, kpt_color): 

        kpt_depth = self.keypoints_depth(depth, kpts)
        _, kpt_proj, _ = self.ego_mapper(kpt_depth)

        preprocessed_depth = self.preprocess_depth(depth)
        agent_view, proj, explored = self.ego_mapper(preprocessed_depth)
        fp_proj = torch.from_numpy(proj).float().unsqueeze(0).unsqueeze(0)
        fp_explored = torch.from_numpy(explored).float().unsqueeze(0).unsqueeze(0)
        pred = torch.hstack((fp_proj, fp_explored)).to(self.device)
        agent_view = self.agent_view.detach_()
        agent_view.fill_(0.)

        x1 = self.map_size // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size // (self.resolution * 2)
        y2 = y1 + self.vision_range


        agent_view[:, :, y1:y2, x1:x2] = pred

        st_pose = current_pose.clone().detach()
        st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution
                                    - self.map_size \
                                    // (self.resolution * 2)) \
                                    / (self.map_size // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose.to(self.device), agent_view.size(), self.device)

        rotated = F.grid_sample(agent_view, rot_mat)
        translated = F.grid_sample(rotated, trans_mat)
        map_mask = translated[0,0].detach()
        color = color_map['white']
        self.tf_map[:,:,0][map_mask >= 0.5] = color[0]
        self.tf_map[:,:,1][map_mask >= 0.5] = color[1]
        self.tf_map[:,:,2][map_mask >= 0.5] = color[2]

        # mark keypoint and line from pose0 to keypoint
        centroid_dist, centroid_angle_deg = self.get_dist_angle(kpt_proj)
        angle_deg_to_centroid = current_pose[0,2] - centroid_angle_deg
        centroid_x, centroid_y = self.pose_at_dist_angle(current_pose[0,0], current_pose[0,1], angle_deg_to_centroid , centroid_dist)
  

        if torch.isnan(centroid_x): 
            return None, None, None, None, None
        self.mark_pose( centroid_x, centroid_y, self.tf_map, 'green')
        self.mark_pose( current_pose[0,0], current_pose[0,1], self.tf_map, kpt_color)

        distance = line_interval(centroid_dist)
        for dist in distance:
            new_x, new_y = self.pose_at_dist_angle(current_pose[0,0], current_pose[0,1], angle_deg_to_centroid , dist)
            self.mark_pose( new_x, new_y, self.tf_map, 'yellow')
        return True, centroid_angle_deg, centroid_dist, centroid_x, centroid_y


    def forward(self, depth0, pose0, depth1, pose1, rgb_match_result):

        # self.tf_map.fill_(0)
        # self.check_map.fill_(0)
        # result0 = False
        # result1 = False 
        # kpt0 = rgb_match_result['mkpts0'][i]
        # kpt1 = rgb_match_result['mkpts1'][i]
        # result0, centroid_angle_deg0, centroid_dist0, keypoint_x0, keypoint_y0 = self.kpt_dist_angle(depth0, kpt0, pose0, 'red')
        # result1, centroid_angle_deg1, centroid_dist1,  keypoint_x1, keypoint_y1  = self.kpt_dist_angle(depth1, kpt1, pose1, 'blue')
        # if result0 and result1: 
        #     if (not torch.isnan(keypoint_x0)) and (not torch.isnan(keypoint_x1)):
        #         return True, keypoint_x0, keypoint_y0 , centroid_angle_deg0, centroid_dist0, keypoint_x1, keypoint_y1, centroid_angle_deg1, centroid_dist1
        
        # return False, keypoint_x0, keypoint_y0 , centroid_angle_deg0, centroid_dist0, keypoint_x1, keypoint_y1, centroid_angle_deg1, centroid_dist1

       
        new_match_result0 = []
        new_match_result1 = []

        valid_once = 0 

        self.tf_map.fill_(0)
        self.check_map.fill_(0)

        list_of_real_diff = []
        list_of_est_diff = []
        landmark = None
        result0 = False
        result1 = False 

        match_list = []
        # print("pose0: ", pose0, " pose1: ", pose1)
        # import pdb; pdb.set_trace()   
        # if len(rgb_match_result['mkpts0'])  > 20: 
        #     random_indices = random.sample(range(len(rgb_match_result['mkpts0'])),2)
        #     new_mkpts0 = [rgb_match_result['mkpts0'][index] for index in random_indices]


        for kpt0, kpt1, mconf in zip(rgb_match_result['mkpts0'], rgb_match_result['mkpts1'], rgb_match_result['mconf']):
            result00, keypoint_angle_deg0, keypoint_dist0, keypoint_x0, keypoint_y0 = self.kpt_dist_angle(depth0, kpt0, pose0, 'red')
            result11, keypoint_angle_deg1, keypoint_dist1,  keypoint_x1, keypoint_y1  = self.kpt_dist_angle(depth1, kpt1, pose1, 'blue')
            if result00 and result11: 
                if (not torch.isnan(keypoint_x0)) and (not torch.isnan(keypoint_x1)):
                    match_list.append([keypoint_x0, keypoint_y0, keypoint_x1, keypoint_y1])
                    new_match_result0.append(kpt0)
                    new_match_result1.append(kpt1)
                    valid_once = 1

        if valid_once :
            match_list_t = torch.tensor(match_list)
            x0 = match_list_t[:,0]
            y0 = match_list_t[:,1]
            x1 = match_list_t[:,2]
            y1 = match_list_t[:,3]

            len_match = x0.shape[0]
            x = x0.reshape(1,-1).t().repeat(1,len_match) - x0.repeat(len_match,1)
            y = y0.reshape(1,-1).t().repeat(1,len_match) - y0.repeat(len_match,1)
            matrix_0 =  (x**2 + y**2 ) **(1/2)


            x = x1.reshape(1,-1).t().repeat(1,len_match) - x1.repeat(len_match,1)
            y = y1.reshape(1,-1).t().repeat(1,len_match) - y1.repeat(len_match,1)
            matrix_1 =  (x**2 + y**2 ) **(1/2)

            matrix_diff = matrix_0 - matrix_1
            matrix_diff[matrix_diff > 0.10] = 0 
            matrix_diff[matrix_diff < -0.10] = 0 

            # matrix_diff[matrix_diff > 0.05] = 0 
            # matrix_diff[matrix_diff < -0.05] = 0 


            filtered_kpt0 = []
            filtered_kpt1 = []
            for idx in range(len_match): 
                if  torch.unique(matrix_diff[idx]).shape[0] > 3: 
                    filtered_kpt0.append(new_match_result0[idx])
                    filtered_kpt1.append(new_match_result1[idx])


            self.tf_map.fill_(0)
            result0, centroid_angle_deg0, centroid_dist0, centroid_x0, centroid_y0 = self.centroid_dist_angle(depth0, np.array(filtered_kpt0), pose0, 'red')
            result1, centroid_angle_deg1, centroid_dist1, centroid_x1, centroid_y1 = self.centroid_dist_angle(depth1, np.array(filtered_kpt1), pose1, 'blue')
            if result0 and result1:
                # print("true")
                return True, centroid_x0, centroid_y0 , centroid_angle_deg0, centroid_dist0, centroid_x1, centroid_y1, centroid_angle_deg1, centroid_dist1
        # return False, centroid_x0, centroid_y0 , centroid_angle_deg0, centroid_dist0, centroid_x1, centroid_y1, centroid_angle_deg1, centroid_dist1
        return False, False, False, False, False, False, False, False, False

        #         landmark = [step0, step1, centroid_angle_deg0.item(), centroid_dist0, \
        #                         centroid_angle_deg1.item(), centroid_dist1, len(filtered_kpt0), new_x0.item(), new_y0.item(), \
        #                         new_x1.item(), new_y1.item()]
        #         self.landmark_list.append(landmark)


        # match_vis = (self.tf_map * 255).detach().cpu().numpy().astype(np.uint8)

        # if (result0 and result1) and (np.nonzero(match_vis)[0].shape[0] > 0 ):
        #     xmin = np.nonzero(match_vis)[0].min()
        #     xmax = np.nonzero(match_vis)[0].max()
        #     ymin = np.nonzero(match_vis)[1].min()
        #     ymax = np.nonzero(match_vis)[1].max()
            
        #     cropped_match_vis = match_vis[xmin: xmax, ymin:ymax]

        #     if (ymax - ymin) < (xmax - xmin): 
        #         width = (xmax - xmin)
        #         canvas = np.zeros((width+10, width+10, 3))
        #         canvas[5:(width+5),5:(ymax-ymin+5),:] = cropped_match_vis
        #     else: 
        #         width = (ymax - ymin)
        #         canvas = np.zeros((width+10, width+10, 3))
        #         canvas[5:(xmax - xmin+5),5:(width+5),:] = cropped_match_vis

        #     match_vis = canvas.astype(np.uint8)

        # if (result0 and result1) == True and landmark == None: 
        #     import pdb; pdb.set_trace()

        # return (result0 and result1) , self.landmark_list, match_vis, landmark
