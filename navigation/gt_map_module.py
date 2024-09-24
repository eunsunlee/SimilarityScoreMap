import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

from utils.model import get_grid, ChannelPool
import matplotlib.pyplot as plt
import numpy as np 


def convert_gray_to_rgb(img):
    if len(img.shape) == 3: 
        img = np.reshape(img, (img.shape[1], img.shape[2]))
    img = (img*255).astype(np.uint8)
    img = np.stack((img, img, img), axis = -1)
    return img

def view_img(image):
    plt.imshow(image)
    plt.show()

class GT_MAP_Module(nn.Module):

    def __init__(self, args):
        super(GT_MAP_Module, self).__init__()

        self.device = args.device
        self.resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.vision_range = args.vision_range

        self.pool = ChannelPool(1)

        self.agent_view = torch.zeros(args.num_processes, 2,
                                      self.map_size_cm // self.resolution,
                                      self.map_size_cm // self.resolution
                                      ).float().to(self.device)
        self.num_scenes = args.num_processes

        


    def forward(self, gt_proj, gt_exp, poses, maps, explored, current_poses):

        pred = torch.zeros((self.num_scenes,2,self.vision_range,self.vision_range))
        for i in range(self.num_scenes):
            pred[i] = torch.stack((gt_proj[i], gt_exp[i]))

        # temp = pred[0,0].detach().cpu().numpy()
        # plt.imsave("temp0.png", temp) 
        # temp = pred[0,1].detach().cpu().numpy()
        # plt.imsave("temp1.png", temp) 
        # temp = pred[1,0].detach().cpu().numpy()
        # plt.imsave("temp2.png", temp) 
        # temp = pred[1,1].detach().cpu().numpy()
        # plt.imsave("temp3.png", temp) 
        # pred = torch.vstack((gt_proj, gt_exp))
        # import pdb; pdb.set_trace()

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

        maps2 = torch.cat((maps.unsqueeze(1),
                            translated[:, :1, :, :]), 1)
        explored2 = torch.cat((explored.unsqueeze(1),
                                translated[:, 1:, :, :]), 1)

        map_pred = self.pool(maps2).squeeze(1)
        exp_pred = self.pool(explored2).squeeze(1)

       
        return map_pred, exp_pred, current_poses





# Neural SLAM Module code
class Pose_Est_Module(nn.Module):
    """
    """

    def __init__(self, args):
        super(Pose_Est_Module, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.use_pe = args.use_pose_estimation
        self.num_processes = args.num_processes

        # Visual Encoding
        resnet = models.resnet18(pretrained=args.pretrained_resnet)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(512, 64, (1, 1), stride=(1, 1)),
            nn.ReLU()
        ]))

        # convolution output size
        input_test = torch.randn(1,
                                 self.n_channels,
                                 self.screen_h,
                                 self.screen_w)
        conv_output = self.conv(self.resnet_l5(input_test))

        self.pool = ChannelPool(1)
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layer
        self.proj1 = nn.Linear(self.conv_output_size, 1024)
        self.proj2 = nn.Linear(1024, 4096)

        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
            self.dropout2 = nn.Dropout(self.dropout)

        # Deconv layers to predict map
        self.deconv = nn.Sequential(*filter(bool, [
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, (4, 4), stride=(2, 2), padding=(1, 1)),
        ]))

        # Pose Estimator
        self.pose_conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(4, 64, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 16, (3, 3), stride=(1, 1)),
            nn.ReLU()
        ]))

        pose_conv_output = self.pose_conv(torch.randn(1, 4,
                                                      self.vision_range,
                                                      self.vision_range))
        self.pose_conv_output_size = pose_conv_output.view(-1).size(0)

        # projection layer
        self.pose_proj1 = nn.Linear(self.pose_conv_output_size, 1024)
        self.pose_proj2_x = nn.Linear(1024, 128)
        self.pose_proj2_y = nn.Linear(1024, 128)
        self.pose_proj2_o = nn.Linear(1024, 128)
        self.pose_proj3_x = nn.Linear(128, 1)
        self.pose_proj3_y = nn.Linear(128, 1)
        self.pose_proj3_o = nn.Linear(128, 1)

        if self.dropout > 0:
            self.pose_dropout1 = nn.Dropout(self.dropout)

        self.st_poses_eval = torch.zeros(args.num_processes,
                                         3).to(self.device)
        self.st_poses_train = torch.zeros(args.slam_batch_size,
                                          3).to(self.device)

        grid_size = self.vision_range * 2
        self.grid_map_eval = torch.zeros(args.num_processes, 2,
                                         grid_size, grid_size
                                         ).float().to(self.device)
        self.grid_map_train = torch.zeros(args.slam_batch_size, 2,
                                          grid_size, grid_size
                                          ).float().to(self.device)

        self.agent_view = torch.zeros(args.num_processes, 2,
                                      self.map_size_cm // self.resolution,
                                      self.map_size_cm // self.resolution
                                      ).float().to(self.device)

    def forward(self, last_projs, last_exp, projs, exp, obs_last, obs, poses, maps, explored, current_poses,
            build_maps=True):


        pred = torch.zeros((self.num_processes ,2,self.vision_range,self.vision_range))
        for i in range(self.num_processes):
            pred[i] = torch.stack((projs[i], exp[i]))
        pred = pred.to(self.device)


        with torch.no_grad(): 

            pred_last = torch.zeros((self.num_processes ,2,self.vision_range,self.vision_range))
            for i in range(self.num_processes):
                pred_last[i] = torch.stack((last_projs[i], last_exp[i]))

            # ST of proj
            vr = self.vision_range
            grid_size = vr * 2

            if build_maps:
                st_poses = self.st_poses_eval.detach_()
                grid_map = self.grid_map_eval.detach_()
            else:
                st_poses = self.st_poses_train.detach_()
                grid_map = self.grid_map_train.detach_()

            st_poses.fill_(0.)
            st_poses[:, 0] = poses[:, 1] * 200. / self.resolution / grid_size
            st_poses[:, 1] = poses[:, 0] * 200. / self.resolution / grid_size
            st_poses[:, 2] = poses[:, 2] * 57.29577951308232
            rot_mat, trans_mat = get_grid(st_poses,
                                          (self.num_processes, 2, grid_size, grid_size),
                                          self.device)

            grid_map.fill_(0.)
            grid_map[:, :, vr:, int(vr / 2):int(vr / 2 + vr)] = pred_last
            translated = F.grid_sample(grid_map, trans_mat)
            rotated = F.grid_sample(translated, rot_mat)
            rotated = rotated[:, :, vr:, int(vr / 2):int(vr / 2 + vr)]

            pred_last_st = rotated


        # Pose estimator
        pose_est_input = torch.cat((pred.detach(), pred_last_st.detach()), dim=1)
        pose_conv_output = self.pose_conv(pose_est_input)
        pose_conv_output = pose_conv_output.view(-1,
                                                 self.pose_conv_output_size)

        proj1 = nn.ReLU()(self.pose_proj1(pose_conv_output))

        if self.dropout > 0:
            proj1 = self.pose_dropout1(proj1)

        proj2_x = nn.ReLU()(self.pose_proj2_x(proj1))
        pred_dx = self.pose_proj3_x(proj2_x)

        proj2_y = nn.ReLU()(self.pose_proj2_y(proj1))
        pred_dy = self.pose_proj3_y(proj2_y)

        proj2_o = nn.ReLU()(self.pose_proj2_o(proj1))
        pred_do = self.pose_proj3_o(proj2_o)

        pose_pred = torch.cat((pred_dx, pred_dy, pred_do), dim=1)
        if self.use_pe == 0:
            pose_pred = pose_pred * self.use_pe

        if build_maps:
            # Aggregate egocentric map prediction in the geocentric map
            # using the predicted pose
            agent_view = self.agent_view.detach_()
            agent_view.fill_(0.)

            x1 = self.map_size_cm // (self.resolution * 2) \
                    - self.vision_range // 2
            x2 = x1 + self.vision_range
            y1 = self.map_size_cm // (self.resolution * 2)
            y2 = y1 + self.vision_range
            agent_view[:, :, y1:y2, x1:x2] = pred

            corrected_pose = poses + pose_pred

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
                                                corrected_pose)

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

            maps2 = torch.cat((maps.unsqueeze(1),
                                translated[:, :1, :, :]), 1)
            explored2 = torch.cat((explored.unsqueeze(1),
                                    translated[:, 1:, :, :]), 1)

            map_pred = self.pool(maps2).squeeze(1)
            exp_pred = self.pool(explored2).squeeze(1)

        else:
            map_pred = None
            exp_pred = None
            current_poses = None

        return map_pred, exp_pred,\
               pose_pred, current_poses


class GT_MAP_Module2(nn.Module):

    def __init__(self, args):
        super(GT_MAP_Module2, self).__init__()

        self.device = args.device
        self.resolution = args.map_resolution
        self.map_size_cm = 2400
        self.vision_range = args.vision_range

        self.pool = ChannelPool(1)

        num_processes = 1
        self.agent_view = torch.zeros(num_processes, 2,
                                      self.map_size_cm // self.resolution,
                                      self.map_size_cm // self.resolution
                                      ).float().to(self.device)

        self.num_scenes = num_processes



    def forward(self, gt_proj, gt_exp, poses, maps, explored, current_poses, b_selected_f):


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

        
        return map_pred, exp_pred, current_poses
