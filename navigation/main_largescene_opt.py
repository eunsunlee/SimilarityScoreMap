import time
from collections import deque

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import gym
import logging
from arguments import get_args
from env import make_vec_envs_largescene
from utils.storage import GlobalRolloutStorage, FIFOMemory
from utils.optimization import get_optimizer
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module, Neural_SLAM_Module_FM
from gt_map_module import GT_MAP_Module, Pose_Est_Module


import algo

import sys
import matplotlib

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from utils.graph import GraphMap, Graph_Mapper
from utils.image_encoder import get_image_feat
from utils.frontiers_opt import get_frontiers_multi, get_frontier_based_global_goals_multi, get_frontiers_multi_ff, get_frontier_based_global_goals_multi_opt
import random

args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

from utils.multi_optimization import multi_agent_optimization


def main():

    filename = time.strftime("runs/%Y%m%d-%H%M%S_" + args.exp_name)
    writer = SummaryWriter(filename)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    policy_loss = 0

    best_cost = 100000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)

    g_masks = torch.ones(num_scenes).float().to(device)
    l_masks = torch.zeros(num_scenes).float().to(device)

    best_local_loss = np.inf
    best_g_reward = -np.inf

    if args.eval:
        traj_lengths = args.max_episode_length // args.num_local_steps
        explored_area_log = np.zeros((num_scenes, num_episodes, traj_lengths))
        explored_ratio_log = np.zeros((num_scenes, num_episodes, traj_lengths))
        combined_explored_ratio_log = np.zeros((1, num_episodes, traj_lengths))
        combined_explored_area_log = np.zeros((1, num_episodes, traj_lengths))

    g_episode_rewards = deque(maxlen=1000)

    l_action_losses = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)

    # Initialize map variables
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations

    torch.set_grad_enabled(False)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, 4, map_size, map_size).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)

    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        if num_scenes == 2: 
            full_pose[1] = torch.tensor([30, 30, 10])
        if num_scenes == 3: 
            full_pose[1] = torch.tensor([30, 30, 10])
            full_pose[2] = torch.tensor([32, 32, 20])
        if num_scenes == 5:
            full_pose[1] = torch.tensor([30, 30, 10])
            full_pose[2] = torch.tensor([32, 32, 20])
            full_pose[3] = torch.tensor([34, 34, 30])
            full_pose[4] = torch.tensor([36, 36, 40])


        planner_pose_inputs[:, :3] = full_pose.cpu().numpy()

        locs = full_pose.cpu().numpy()
        for e in range(num_scenes):
            loc_r, loc_c = [int(locs[e, 1] * 100.0 / args.map_resolution), int(locs[e, 0] * 100.0 / args.map_resolution)]
            full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0
            planner_pose_inputs[e, 3:] = [0, map_size, 0, map_size]

    init_map_and_pose()

    # Local policy observation space
    l_observation_space = gym.spaces.Box(0, 255,
                                         (3,
                                          args.frame_width,
                                          args.frame_width), dtype='uint8')

    # Local and Global policy recurrent layer sizes
    l_hidden_size = args.local_hidden_size
    g_hidden_size = args.global_hidden_size

    gt_map_module = GT_MAP_Module(args).to(device)

    # slam
    nslam_module = Neural_SLAM_Module_FM(args).to(device)
    slam_optimizer = get_optimizer(nslam_module.parameters(),
                                   args.slam_optimizer)

    # ------------------------ Load pose estimator ----------------------------- #
    pose_est_module = Pose_Est_Module(args).to(device)
    state_dict = torch.load(args.load_slam, map_location=lambda storage, loc: storage)
    pose_est_module.load_state_dict(state_dict)
    pose_est_module.eval()
    # ------------------------------------------------------------------------- #

    envs = make_vec_envs_largescene(args, 0, 0)

    # Local policy
    l_policy = Local_IL_Policy(l_observation_space.shape, envs.action_space.n,
                               recurrent=args.use_recurrent_local,
                               hidden_size=l_hidden_size,
                               deterministic=args.use_deterministic_local).to(device)

    # Loading model
    if args.load_slam != "0":
        print("Loading slam {}".format(args.load_slam))
        state_dict = torch.load(args.load_slam,
                                map_location=lambda storage, loc: storage)
        nslam_module.load_state_dict(state_dict)
    nslam_module.eval()



    if args.load_local != "0":
        print("Loading local {}".format(args.load_local))
        state_dict = torch.load(args.load_local,
                                map_location=lambda storage, loc: storage)
        l_policy.load_state_dict(state_dict)
    l_policy.eval()

    ep_iter = 0 
    for ep_num in range(num_episodes):
        envs = make_vec_envs_largescene(args, ep_num, args.add_ind)

        obs, depth, infos = envs.reset()


        total_num_steps = -1
        g_reward = 0

        # Predict map from frame 1:
        gt_pose = torch.from_numpy(np.asarray(
            [infos[env_idx]['gt_pose'] for env_idx
            in range(num_scenes)])
        ).float().to(device)

        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
            in range(num_scenes)])
        ).float().to(device)

        gt_fp_projs = torch.from_numpy(np.asarray(
            [infos[env_idx]['fp_proj'] for env_idx 
            in range(num_scenes)])
        ).float().to(device)

        gt_fp_explored = torch.from_numpy(np.asarray(
            [infos[env_idx]['fp_explored'] for env_idx
            in range(num_scenes)])
        ).float().to(device)

        obs_rgb = torch.from_numpy(np.asarray(
            [infos[env_idx]['obs_rgb'] for env_idx
            in range(num_scenes)])
        ).float().to(device)

        gt_fp_depth = torch.from_numpy(np.asarray(
            [infos[env_idx]['obs_depth'] for env_idx
            in range(num_scenes)])
        ).float().to(device)

        full_map[:, 0, :, :], full_map[:, 1, :, :], full_pose= \
            gt_map_module(gt_fp_projs, gt_fp_explored, poses, full_map[:, 0, :, :],
                full_map[:, 1, :, :], full_pose)  

        global_input = torch.zeros(num_scenes, 8, map_size, map_size)
        

        locs = full_pose.cpu().numpy()
        for e in range(num_scenes): 
            loc_r, loc_c = [int(locs[e, 1] * 100.0 / args.map_resolution), int(locs[e, 0] * 100.0 / args.map_resolution)]
            full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        global_input[:, 0:4, :, :] = full_map[:, 0:4, :, :].detach()
        global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(full_map[:, 0:4, :, :])

        x_goal = random.sample(range(0,map_size),num_scenes)
        y_goal = random.sample(range(0,map_size),num_scenes)
        global_goals = []
        for e in range(num_scenes): 
            global_goals.append([x_goal[e], y_goal[e]])

        # Compute planner inputs
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['goal'] = global_goals[e]
            p_input['map_pred'] = global_input[e, 0, :, :].detach().cpu().numpy()
            p_input['exp_pred'] = global_input[e, 1, :, :].detach().cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]

        # Output stores local goals as well as the the ground-truth action
        output = envs.get_short_term_goal(planner_inputs)

        last_proj = gt_fp_projs.detach().clone()
        last_exp = gt_fp_explored.detach().clone()
        last_obs = obs.detach()
        local_rec_states = torch.zeros(num_scenes, l_hidden_size).to(device)
        start = time.time()


        torch.set_grad_enabled(False)


        # Initialize graph maps 
        pose_graphs = []
        for i in range(num_scenes): 
            pose_graph = GraphMap(device)
            pose_graphs.append(pose_graph)
        # --------------------  Get image features   -------------------------         
        img_feats = []
        for i in range(num_scenes): 
            img_feat = get_image_feat(obs_rgb[i:i+1]).cpu().detach().numpy()
            img_feats.append(img_feat)

        # --------------------------------------------------------------------

        # -------------------  Update GraphMap node  -------------------------
        for i in range(num_scenes): 
            pose_graphs[i].add_robot_node(obs_rgb[i].detach().cpu().numpy(), \
                gt_fp_depth[i].detach().cpu().numpy(), full_pose[i:i+1].detach().cpu(), \
                gt_fp_projs[i:i+1].detach().cpu(), gt_fp_explored[i:i+1].detach().cpu(), \
                img_feats[i]) 
            pose_graphs[i].add_rep_node(0)
        # ------------------------------------------------------------------



        for step in range(args.max_episode_length):
            if step == 0: 
                map_combined = 0 
                map_combined_envs = [0]*num_scenes
                map_combined_envs[0] = 1 
                prev_goals = [[]]*num_scenes
                combined_prev_explored_area = 0 
                for n in range(len(infos)): 
                    temp = "scene ",n, " : " , infos[n]['scene_name'].split('/')[-1], ", sp: ", infos[n]['start_position']
                prev_explored_area_list = [0]*len(infos)
                reset_infos = infos
                print(" area size: ", reset_infos[0]['explorable_map'].sum())
                temp = "scene : {}, scene_name: {}, scene_size: {} ".format(ep_num,infos[n]['scene_name'].split('/')[-1].split('.')[0],int(reset_infos[0]['explorable_map'].sum()) )
                logging.info(temp)

                if args.eval:
                    traj_lengths = args.max_episode_length // args.num_local_steps
                    ep_combined_explored_ratio_log = np.zeros(traj_lengths)
                    ep_combined_explored_area_log = np.zeros(traj_lengths)


            total_num_steps += 1

            g_step = (step // args.num_local_steps) % args.num_global_steps
            eval_g_step = step // args.num_local_steps + 1
            l_step = step % args.num_local_steps

            # ------------------------------------------------------------------
            # Local Policy
            del last_obs
            last_obs = obs.detach()
            local_masks = l_masks
            local_goals = output[:, :-1].to(device).long()

            if args.train_local:
                torch.set_grad_enabled(True)

            action, action_prob, local_rec_states = l_policy(
                obs,
                local_rec_states,
                local_masks,
                extras=local_goals,
            )

            if args.train_local:
                action_target = output[:, -1].long().to(device)
                policy_loss += nn.CrossEntropyLoss()(action_prob, action_target)
                torch.set_grad_enabled(False)
            l_action = action.cpu()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Env step
            obs, depth, rew, done, infos = envs.step(l_action)

            l_masks = torch.FloatTensor([0 if x else 1
                                         for x in done]).to(device)
            g_masks *= l_masks
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
                
            if step == args.max_episode_length - 3 : 
                for e in range(num_scenes): 
                    plt.imsave("{}temp_{}_{}.png".format(dump_dir, ep_num, e), full_map[e,0].cpu().numpy() )
                
            if step == args.max_episode_length - 1 :  # Last episode step
                print(" end of episode")
            
                log = "\nEp Combined Exp Ratio: \n"

                for i in range(ep_combined_explored_ratio_log.shape[0]):
                    log += "{:.5f}, ".format(ep_combined_explored_ratio_log[i])

                log += "\nEp Combined Exp Area: \n"
                for i in range(ep_combined_explored_area_log.shape[0]):
                    log += "{:.5f}, ".format(ep_combined_explored_area_log[i])
                print(log)
                logging.info(log)
                if np.max(map_combined_envs[1:]): 
                    temp = "================================ end of episode ==== optimized ======" + str(map_combined_envs) + "==========="
                    logging.info(temp)
                else: 
                    logging.info("================================ end of episode ===============================")

                init_map_and_pose()

                del last_obs
                last_obs = obs.detach()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Neural SLAM Module
            gt_pose = torch.from_numpy(np.asarray(
                [infos[env_idx]['gt_pose'] for env_idx
                in range(num_scenes)])
            ).float().to(device)

            poses = torch.from_numpy(np.asarray(
                [infos[env_idx]['sensor_pose'] for env_idx
                 in range(num_scenes)])
            ).float().to(device)

            gt_fp_projs = torch.from_numpy(np.asarray(
                [infos[env_idx]['fp_proj'] for env_idx 
                in range(num_scenes)])
            ).float().to(device)

            gt_fp_explored = torch.from_numpy(np.asarray(
                [infos[env_idx]['fp_explored'] for env_idx
                in range(num_scenes)])
            ).float().to(device)

            obs_rgb = torch.from_numpy(np.asarray(
                [infos[env_idx]['obs_rgb'] for env_idx
                in range(num_scenes)])
            ).float().to(device)

            gt_fp_depth = torch.from_numpy(np.asarray(
                [infos[env_idx]['obs_depth'] for env_idx
                in range(num_scenes)])
            ).float().to(device)
            
            full_map[:, 0, :, :], full_map[:, 1, :, :], full_pose= \
                gt_map_module(gt_fp_projs, gt_fp_explored, poses, full_map[:, 0, :, :],
                    full_map[:, 1, :, :], full_pose)     

            locs = full_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs
            full_map[:, 2, :, :].fill_(0.)
            for e in range(num_scenes): 
                loc_r, loc_c = [int(locs[e, 1] * 100.0 / args.map_resolution), int(locs[e, 0] * 100.0 / args.map_resolution)]
                full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            last_proj = gt_fp_projs.detach().clone()
            last_exp = gt_fp_explored.detach().clone()
            last_obs = obs.detach().clone()
            # ------------------------------------------------------------------

            # --------------------  Get image features   -------------------------         
            img_feats = []
            for i in range(num_scenes): 
                img_feat = get_image_feat(obs_rgb[i:i+1]).cpu().detach().numpy()
                img_feats.append(img_feat)

            # --------------------------------------------------------------------


            # -------------------  Update GraphMap node  -------------------------
            for i in range(num_scenes): 
                pose_graphs[i].add_robot_node(obs_rgb[i].detach().cpu().numpy(), \
                    gt_fp_depth[i].detach().cpu().numpy(), full_pose[i:i+1].detach().cpu(), \
                    gt_fp_projs[i:i+1].detach().cpu(), gt_fp_explored[i:i+1].detach().cpu(), \
                    img_feats[i]) 

                pose_graphs[i].add_robot_edge(pose_graphs[i].node_at(step), \
                    pose_graphs[i].node_at(step+1), \
                    (poses[i][0].item(), poses[i][1].item(), poses[i][2].item()), \
                    (gt_pose[i][0].item(), gt_pose[i][1].item(), gt_pose[i][2].item()), 0) 
            # --------------------------------------------------------------------

            # -------------------  Update rep. node  -------------------------
            for i in range(num_scenes): 
                last_rep_node_id = max(list(pose_graphs[i].rep_nodes))
                last_rep_node_feat = pose_graphs[i].node_at(last_rep_node_id).img_feat

                logits = np.matmul(last_rep_node_feat, img_feats[i].transpose(1,0))
                if logits < 0.95 or last_rep_node_id == 0 : # add a new rep node 
                    pose_graphs[i].add_rep_node(step+1)
                else: # update last rep node to the most recent node 
                    pose_graphs[i].rep_nodes.remove(last_rep_node_id)
                    pose_graphs[i].add_rep_node(step+1)

            # --------------------------------------------------------------------


            pose_position = full_pose[:,:2].detach().clone().cpu().numpy() * 100. / args.map_resolution
            dist_to_frontier = np.linalg.norm(global_goals - pose_position, axis=1)

            # ------------------------------------------------------------------

            if l_step == args.num_local_steps - 1 or np.min(dist_to_frontier) < 30:
                for e in range(num_scenes): 
                    if map_combined_envs[e] == 0: 
                        input_pose_graphs = []
                        input_pose_graphs.append(pose_graphs[0])
                        input_pose_graphs.append(pose_graphs[e])
                        input_full_map = torch.zeros(2, 4, map_size, map_size).float().to(device)
                        input_full_map[0] = full_map[0].detach().clone()
                        input_full_map[1] = full_map[e].detach().clone()
                        input_full_pose = torch.zeros(2, 3).float().to(device)
                        input_full_pose[0] = full_pose[0].detach().clone()
                        input_full_pose[1] = full_pose[e].detach().clone()

                        map_combined, combined_map, new_full_pose, new_pose_graphs, remapped_full_map = multi_agent_optimization(args, 2, input_pose_graphs, input_full_map, input_full_pose, device)
                        map_combined_envs[e] = map_combined
                        if map_combined: 
                    
                            pose_graphs[e] = new_pose_graphs[1]

                            tmp_map_module = Graph_Mapper(args).to(device)
                            tmp_full_pose = torch.zeros(1, 3).float().to(device)
                            tmp_full_map = torch.zeros(1, 4, map_size, map_size).float().to(device)
                            freq_map = torch.zeros(1, 4, map_size, map_size).float().to(device)
                            tmp_full_pose[0,:] =  torch.tensor(pose_graphs[e].node_at(0).pose).to(device)

                            locs = tmp_full_pose.cpu().numpy()
                            loc_r, loc_c = [int(locs[0, 1] * 100.0 / args.map_resolution), int(locs[0, 0] * 100.0 / args.map_resolution)]
                            tmp_full_map[0, 3, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

                            tmp_full_map[:,3, ]
                            for i in range(1,len(pose_graphs[e].nodes)):
                                tmp_pose = torch.tensor(pose_graphs[e].edge_at(i-1, i).sensor_pose).unsqueeze(0).to(device)
                                tmp_proj = pose_graphs[e].node_at(i).proj_map.to(device)
                                tmp_exp = pose_graphs[e].node_at(i).exp_map.to(device)

                                tmp_full_map[:, 0, :, :], tmp_full_map[:, 1, :, :], tmp_full_pose, freq_map = \
                                    tmp_map_module(tmp_proj, tmp_exp, tmp_pose, tmp_full_map[:, 0, :, :],
                                        tmp_full_map[:, 1, :, :], tmp_full_pose, 1, freq_map)  
                                locs = tmp_full_pose.cpu().numpy()
                                loc_r, loc_c = [int(locs[0, 1] * 100.0 / args.map_resolution), int(locs[0, 0] * 100.0 / args.map_resolution)]
                                tmp_full_map[0, 3, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

                            full_map[e] = tmp_full_map[0].detach().clone()
                            len_nodes = len(pose_graphs[e].nodes)
                            full_pose[e] = torch.tensor(pose_graphs[e].node_at(len_nodes-1).pose)
    
                            locs = full_pose.cpu().numpy()
                            loc_r, loc_c = [int(locs[e, 1] * 100.0 / args.map_resolution), int(locs[e, 0] * 100.0 / args.map_resolution)]
                            full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

                # generate image features 
                agent_img_feats = []
                len_each_agent = []
                
                for e in range(num_scenes): 
                    img_feat_per_agent = np.zeros((len(pose_graphs[e].nodes), 512))
                    len_each_agent.append(len(pose_graphs[e].nodes))
                    for i, node in enumerate(pose_graphs[e].nodes):
                        img_feat_per_agent[i:i+1] = node.img_feat
                    agent_img_feats.append(img_feat_per_agent)

                all_agents_arr = np.concatenate(agent_img_feats, axis = 0 )
                least_sim_node_poses = []
                list_sum_of_logits = []
                for e in range(num_scenes): 
                    agent_logits = np.matmul(agent_img_feats[e], all_agents_arr.transpose(1,0))
                    cum_len = 0 
                    for o in range(num_scenes): 
                        cum_len += len_each_agent[o]
                        if e != o: 
                            agent_logits[:,cum_len-1] = agent_logits[:,cum_len-1]*1.1
                    sum_of_logits = np.sum(agent_logits, axis=1)
                    list_sum_of_logits.append(sum_of_logits)
                    least_sim_index = np.argmin(sum_of_logits)
                    selected_node = pose_graphs[e].node_at(least_sim_index)
                    selected_node_pose = selected_node.pose
                    least_sim_node_poses.append(selected_node_pose[0])


                # generate a frequency map for all agents 
                freq_map_by_node_list = []
                for e in range(num_scenes):
                    freq_map_by_node = torch.zeros(1, 4, map_size, map_size).float().to(device)

                    tmp_map_module = Graph_Mapper(args).to(device)
                    tmp_full_pose = torch.zeros(1, 3).float().to(device)
                    tmp_full_pose[0,0] = (args.map_size_cm / 100.0 / 2.0)
                    tmp_full_pose[0,1] = (args.map_size_cm / 100.0 / 2.0)
                    tmp_full_map = torch.zeros(1, 4, map_size, map_size).float().to(device)

                    for i in range(1,step):
                        tmp_pose = torch.tensor(pose_graphs[e].edge_at(i-1, i).gt_sensor_pose).unsqueeze(0).to(args.device)
                        tmp_proj = pose_graphs[e].node_at(i).proj_map.to(args.device)
                        tmp_exp = pose_graphs[e].node_at(i).exp_map.to(args.device)
                        tmp_sim_score = list_sum_of_logits[e][i]

                        b_selected_f = 1 

                        tmp_full_map[:, 0, :, :], tmp_full_map[:, 1, :, :], tmp_full_pose, freq_map_by_node = \
                            tmp_map_module(tmp_proj, tmp_exp, tmp_pose, tmp_full_map[:, 0, :, :],
                                tmp_full_map[:, 1, :, :], tmp_full_pose, b_selected_f, freq_map_by_node, tmp_sim_score)  
                    
                    freq_map_by_node_list.append(freq_map_by_node)
                locs = full_pose.cpu().numpy()

                global_input[:, 0:4, :, :] = full_map[:, 0:4, :, :]
                global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(full_map[:, 0:4, :, :])   

                frontiers, stats, obstacle_map, obstacle_region, explored_map, frontier_img, freq_vals_list  = get_frontiers_multi(args, full_map.detach().clone(), freq_map_by_node_list, 0, map_combined_envs)

                new_frontiers = []
                new_stats = []
                new_freq_vals_list = [] 
                for e in range(num_scenes):
                    n_frontiers_e = []
                    n_stats_e = []
                    n_freq_vals_list_e = []
                    for f in range(frontiers[e].shape[0]):
                        _frontier = frontiers[e][f].tolist()
                        _stats = stats[e][f].tolist()
                        _freq_vals = freq_vals_list[e][f]

                        if _frontier not in prev_goals[e] : 
                            n_frontiers_e.append(_frontier)
                            n_stats_e.append(_stats)
                            n_freq_vals_list_e.append(_freq_vals)
                    new_frontiers.append(np.array(n_frontiers_e))
                    new_stats.append(np.array(n_stats_e))
                    new_freq_vals_list.append(np.array(n_freq_vals_list_e))

                frontiers = new_frontiers
                stats = new_stats
                freq_vals_list = new_freq_vals_list



                frontier_len = []
                for e in range(num_scenes): 
                    frontier_len.append(frontiers[e].shape[0])
                
                if 0 in frontier_len: 
                    print("frontier is 0 at : ", np.where(np.array(frontier_len) == 0 ) [0].tolist())
                x_goal = random.sample(range(0,map_size),num_scenes)
                y_goal = random.sample(range(0,map_size),num_scenes)
                temp_global_goals = []
                for e in range(num_scenes): 
                    temp_global_goals.append([x_goal[e], y_goal[e]])


                global_goals = get_frontier_based_global_goals_multi_opt(args, temp_global_goals, frontiers, stats, full_pose, 0, freq_vals_list, least_sim_node_poses, map_combined_envs, full_map.detach().clone(), frontier_img)

                for e in range(num_scenes): 
                    prev_goals[e].append([global_goals[e][1], global_goals[e][0]])

            # ------------------------------------------------------------------


            if l_step == args.num_local_steps -1 : 

                for e in range(len(infos)):
                    explored_map = infos[e]['explored_map']
                    explorable_map = reset_infos[e]['explorable_map']
                    # plt.imsave("temp_{}.png".format(e), explorable_map)
                    curr_explored = explored_map*explorable_map
                    curr_explored_area = curr_explored.sum()

                    reward_scale = explorable_map.sum()
                    m_reward = (curr_explored_area - prev_explored_area_list[e])*1.
                    m_ratio = m_reward/reward_scale
                    m_reward = m_reward * 25./10000. # converting to m^2
                    prev_explored_area_list[e] = curr_explored_area

                    m_reward *= 0.02 # Reward Scaling
                    infos[e]['exp_reward'] = m_reward
                    infos[e]['exp_ratio'] = m_ratio

                # get combined exp ratio 
                explorable_map = reset_infos[0]['explorable_map']
                explored_map = infos[0]['explored_map']
                for e in range(num_scenes): 
                    if e != 0 : 
                        explored_map += infos[e]['explored_map']
                explored_map[explored_map > 0 ] = 1.
                curr_explored = explored_map*explorable_map
                curr_explored_area = curr_explored.sum()

                reward_scale = explorable_map.sum()
                combined_m_reward = (curr_explored_area - combined_prev_explored_area)*1.
                combined_m_ratio = combined_m_reward/reward_scale
                combined_m_reward = combined_m_reward * 25./10000. # converting to m^2

                combined_prev_explored_area = curr_explored_area


                # Get exploration reward and metrics
                g_reward = torch.from_numpy(np.asarray([infos[env_idx]['exp_reward'] for env_idx in range(num_scenes)])).float().to(device)

                if args.eval:
                    g_reward = g_reward*50.0 # Convert reward to area in m2

                g_process_rewards += g_reward.cpu().numpy()
                g_total_rewards = g_process_rewards * (1 - g_masks.cpu().numpy())
                g_process_rewards *= g_masks.cpu().numpy()
                per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

                if np.sum(g_total_rewards) != 0:
                    for tr in g_total_rewards:
                        g_episode_rewards.append(tr) if tr != 0 else None

                if args.eval:
                    exp_ratio = torch.from_numpy(np.asarray([infos[env_idx]['exp_ratio'] for env_idx in range(num_scenes)])).float()
                    combined_exp_ratio = combined_m_ratio 
                    combined_exp_reward = combined_m_reward

                    for e in range(num_scenes):
                        explored_area_log[e, ep_num, eval_g_step - 1] = \
                            explored_area_log[e, ep_num, eval_g_step - 2] + \
                            g_reward[e].cpu().numpy()
                        explored_ratio_log[e, ep_num, eval_g_step - 1] = \
                            explored_ratio_log[e, ep_num, eval_g_step - 2] + \
                            exp_ratio[e].cpu().numpy()
                    combined_explored_ratio_log[0, ep_num, eval_g_step - 1] =  combined_explored_ratio_log[0, ep_num, eval_g_step - 2] + combined_exp_ratio
                    combined_explored_area_log[0, ep_num, eval_g_step - 1] =  combined_explored_area_log[0, ep_num, eval_g_step - 2] + combined_exp_reward
                    ep_combined_explored_ratio_log[eval_g_step-1] = ep_combined_explored_ratio_log[eval_g_step -2] + combined_exp_ratio
                    ep_combined_explored_area_log[eval_g_step-1] = ep_combined_explored_area_log[eval_g_step -2] + combined_exp_reward

            # ------------------------------------------------------------------
            # Get short term goal
            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                p_input['map_pred'] = full_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = full_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = planner_pose_inputs[e]
                p_input['goal'] = global_goals[e]

            output = envs.get_short_term_goal(planner_inputs)
            # ------------------------------------------------------------------


    # Print and save model performance numbers during evaluation
    if args.eval:
        logfile = open("{}/explored_area.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_area_log[e].shape[0]):
                logfile.write(str(explored_area_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        logfile = open("{}/explored_ratio.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_ratio_log[e].shape[0]):
                logfile.write(str(explored_ratio_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        log = "Final Exp Area: \n"
        for i in range(explored_area_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_area_log[:, :, i]))

        log += "\nFinal Exp Ratio: \n"
        for i in range(explored_ratio_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_ratio_log[:, :, i]))

        log += "\nCombined Exp Ratio: \n"
        for i in range(combined_explored_ratio_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(combined_explored_ratio_log[:, :, i]))

        log += "\nCombined Exp Area: \n"
        for i in range(combined_explored_area_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(combined_explored_area_log[:, :, i]))
        print(log)
        logging.info(log)



if __name__ == "__main__":
    main()