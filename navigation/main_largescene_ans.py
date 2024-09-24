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
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module

import algo
import sys
import matplotlib

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from utils.exploration_metric import get_exp_ratio
from gt_map_module import GT_MAP_Module, Pose_Est_Module


args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]


def main():
    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))


    filename = time.strftime("runs/%Y%m%d-%H%M%S_" + args.exp_name)
    writer = SummaryWriter(filename)

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
    envs = make_vec_envs_largescene(args, 0, 0)
    obs, depth, infos = envs.reset()
    reset_infos = infos

    # Initialize map variables
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations

    torch.set_grad_enabled(False)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
                       int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, 4, local_w, local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :4, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()

    # Global policy observation space
    g_observation_space = gym.spaces.Box(0, 1,
                                         (8,
                                          local_w,
                                          local_h), dtype='uint8')

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=(2,), dtype=np.float32)

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
    nslam_module = Neural_SLAM_Module(args).to(device)
    slam_optimizer = get_optimizer(nslam_module.parameters(),
                                   args.slam_optimizer)

    # ------------------------ Load pose estimator ----------------------------- #
    pose_est_module = Pose_Est_Module(args).to(device)
    state_dict = torch.load(args.load_slam, map_location=lambda storage, loc: storage)
    pose_est_module.load_state_dict(state_dict)
    pose_est_module.eval()
    # ------------------------------------------------------------------------- #


    # Global policy
    g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'downscaling': args.global_downscaling
                                      }).to(device)
    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.global_lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    # Local policy
    l_policy = Local_IL_Policy(l_observation_space.shape, envs.action_space.n,
                               recurrent=args.use_recurrent_local,
                               hidden_size=l_hidden_size,
                               deterministic=args.use_deterministic_local).to(device)
    local_optimizer = get_optimizer(l_policy.parameters(),
                                    args.local_optimizer)

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      1).to(device)

    slam_memory = FIFOMemory(args.slam_memory_size)

    # Loading model
    if args.load_slam != "0":
        print("Loading slam {}".format(args.load_slam))
        state_dict = torch.load(args.load_slam,
                                map_location=lambda storage, loc: storage)
        nslam_module.load_state_dict(state_dict)

    if not args.train_slam:
        nslam_module.eval()

    if args.load_global != "0":
        print("Loading global {}".format(args.load_global))
        state_dict = torch.load(args.load_global,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if not args.train_global:
        g_policy.eval()

    if args.load_local != "0":
        print("Loading local {}".format(args.load_local))
        state_dict = torch.load(args.load_local,
                                map_location=lambda storage, loc: storage)
        l_policy.load_state_dict(state_dict)

    if not args.train_local:
        l_policy.eval()
    total_num_steps = -1

    eff_list = []
    for ep_num in range(num_episodes):
        temp = " --- ep num : ", ep_num 

        gt_pose = torch.from_numpy(np.asarray([infos[env_idx]['gt_pose'] for env_idx in range(num_scenes)])).float().to(device)
        poses = torch.from_numpy(np.asarray([infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])).float().to(device)
        gt_fp_projs = torch.from_numpy(np.asarray([infos[env_idx]['fp_proj'] for env_idx in range(num_scenes)])).float().to(device)
        gt_fp_explored = torch.from_numpy(np.asarray([infos[env_idx]['fp_explored'] for env_idx in range(num_scenes)])).float().to(device)
        obs_rgb = torch.from_numpy(np.asarray([infos[env_idx]['obs_rgb'] for env_idx in range(num_scenes)])).float().to(device)
        gt_fp_depth = torch.from_numpy(np.asarray([infos[env_idx]['obs_depth'] for env_idx in range(num_scenes)])).float().to(device)

        local_map[:, 0, :, :], local_map[:, 1, :, :], local_pose= \
            gt_map_module(gt_fp_projs, gt_fp_explored, poses, local_map[:, 0, :, :],
                local_map[:, 1, :, :], local_pose)  


        # Compute Global policy input
        locs = local_pose.cpu().numpy()

        global_input = torch.zeros(num_scenes, 8, local_w, local_h)
        global_orientation = torch.zeros(num_scenes, 1).long()

        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
            global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

        global_input[:, 0:4, :, :] = local_map.detach()
        global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(full_map)

        g_rollouts.obs[0].copy_(global_input)
        g_rollouts.extras[0].copy_(global_orientation)

        # Run Global Policy (global_goals = Long-Term Goal)
        g_value, g_action, g_action_log_prob, g_rec_states = \
            g_policy.act(
                g_rollouts.obs[0],
                g_rollouts.rec_states[0],
                g_rollouts.masks[0],
                extras=g_rollouts.extras[0],
                deterministic=False
            )

        cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
        global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                        for action in cpu_actions]

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

        g_reward = 0

        torch.set_grad_enabled(False)
            
        for step in range(args.max_episode_length):
            if step == 0 and ep_num % args.traj_per_scene == 0  : 
                envs = make_vec_envs_largescene(args, ep_num, args.add_ind)
            if step == 0 : 

                obs, depth, infos = envs.reset()
                reset_infos = infos
                prev_explored_area = 0 
                combined_prev_explored_area = 0 
                prev_explored_area_list = [0]*len(infos)

                for n in range(len(infos)): 
                    temp = "scene ",n, " : " , infos[n]['scene_name'].split('/')[-1], ", sp: ", infos[n]['start_position']

                print(" area size: ", reset_infos[0]['explorable_map'].sum())
                temp = "scene : {}, scene_name: {}, scene_size: {} ".format(ep_num,infos[0]['scene_name'].split('/')[-1].split('.')[0],int(reset_infos[0]['explorable_map'].sum()) )
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
            last_proj = gt_fp_projs.detach().clone()
            last_exp = gt_fp_explored.detach().clone()
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


            if step == 5: 
                for n in range(len(infos)): 
                    temp = "scene ",n, " : " , infos[n]['scene_name'].split('/')[-1], ", sp: ", infos[n]['start_position']
                    # logging.info(temp)

            l_masks = torch.FloatTensor([0 if x else 1
                                        for x in done]).to(device)
            g_masks *= l_masks
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Reinitialize variables when episode ends
            if step == args.max_episode_length - 1:  # Last episode step
                print(" end of episode")

                for e in range(num_scenes): 
                    plt.imsave("{}temp_{}_{}.png".format(dump_dir, ep_num, e), full_map[e,0].cpu().numpy() )
                

                log = "\nEp Combined Exp Ratio: \n"

                for i in range(ep_combined_explored_ratio_log.shape[0]):
                    log += "{:.5f}, ".format(ep_combined_explored_ratio_log[i])

                log += "\nEp Combined Exp Area: \n"
                for i in range(ep_combined_explored_area_log.shape[0]):
                    log += "{:.5f}, ".format(ep_combined_explored_area_log[i])
                print(log)
                logging.info(log)
                logging.info("================================ end of episode ===============================")

                init_map_and_pose()
                del last_obs
                last_obs = obs.detach()

            # ------------------------------------------------------------------

            gt_pose = torch.from_numpy(np.asarray([infos[env_idx]['gt_pose'] for env_idx in range(num_scenes)])).float().to(device)
            poses = torch.from_numpy(np.asarray([infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])).float().to(device)
            gt_fp_projs = torch.from_numpy(np.asarray([infos[env_idx]['fp_proj'] for env_idx in range(num_scenes)])).float().to(device)
            gt_fp_explored = torch.from_numpy(np.asarray([infos[env_idx]['fp_explored'] for env_idx in range(num_scenes)])).float().to(device)
            obs_rgb = torch.from_numpy(np.asarray([infos[env_idx]['obs_rgb'] for env_idx in range(num_scenes)])).float().to(device)
            gt_fp_depth = torch.from_numpy(np.asarray([infos[env_idx]['obs_depth'] for env_idx in range(num_scenes)])).float().to(device)

            local_map[:, 0, :, :], local_map[:, 1, :, :], local_pose= \
                gt_map_module(gt_fp_projs, gt_fp_explored, poses, local_map[:, 0, :, :],
                    local_map[:, 1, :, :], local_pose)    

            locs = local_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs + origins
            local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                local_map[e, 2:, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Global Policy
            if l_step == args.num_local_steps - 1:
                # For every global step, update the full and local maps
                for e in range(num_scenes):
                    full_map[e, :4, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                        local_map[e]
                    full_pose[e] = local_pose[e] + \
                                torch.from_numpy(origins[e]).to(device).float()

                    locs = full_pose[e].cpu().numpy()
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]

                    lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                    (local_w, local_h),
                                                    (full_w, full_h))

                    planner_pose_inputs[e, 3:] = lmb[e]
                    origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                                lmb[e][0] * args.map_resolution / 100.0, 0.]

                    local_map[e] = full_map[e, :4,
                                lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
                    local_pose[e] = full_pose[e] - \
                                    torch.from_numpy(origins[e]).to(device).float()

                locs = local_pose.cpu().numpy()
                for e in range(num_scenes):
                    global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
                global_input[:, 0:4, :, :] = local_map
                global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(full_map)

                # Get exploration reward and metrics
                g_reward = torch.from_numpy(np.asarray(
                    [infos[env_idx]['exp_reward'] for env_idx
                    in range(num_scenes)])
                ).float().to(device)

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

                g_reward = torch.from_numpy(np.asarray([infos[env_idx]['exp_reward'] for env_idx in range(num_scenes)])).float().to(device)


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


                if args.eval:
                    g_reward = g_reward*50.0 # Convert reward to area in m2

                g_process_rewards += g_reward.cpu().numpy()
                g_total_rewards = g_process_rewards * \
                                (1 - g_masks.cpu().numpy())
                g_process_rewards *= g_masks.cpu().numpy()
                per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

                if np.sum(g_total_rewards) != 0:
                    for tr in g_total_rewards:
                        g_episode_rewards.append(tr) if tr != 0 else None

                
                if args.eval:
                    exp_ratio = torch.from_numpy(np.asarray(
                        [infos[env_idx]['exp_ratio'] for env_idx
                        in range(num_scenes)])
                    ).float()
                    combined_exp_ratio = combined_m_ratio
                    combined_exp_reward = combined_m_reward

                    for e in range(num_scenes):
                        explored_area_log[e, ep_num, eval_g_step - 1] =  explored_area_log[e, ep_num, eval_g_step - 2] +  g_reward[e].cpu().numpy()
                        explored_ratio_log[e, ep_num, eval_g_step - 1] =  explored_ratio_log[e, ep_num, eval_g_step - 2] +  exp_ratio[e].cpu().numpy()
                    combined_explored_ratio_log[0, ep_num, eval_g_step - 1] =  combined_explored_ratio_log[0, ep_num, eval_g_step - 2] + combined_exp_ratio
                    combined_explored_area_log[0, ep_num, eval_g_step - 1] =  combined_explored_area_log[0, ep_num, eval_g_step - 2] + combined_exp_reward
                    ep_combined_explored_ratio_log[eval_g_step-1] = ep_combined_explored_ratio_log[eval_g_step -2] + combined_exp_ratio
                    ep_combined_explored_area_log[eval_g_step-1] = ep_combined_explored_area_log[eval_g_step -2] + combined_exp_reward

                # Add samples to global policy storage
                g_rollouts.insert(
                    global_input, g_rec_states,
                    g_action, g_action_log_prob, g_value,
                    g_reward, g_masks, global_orientation
                )

                # Sample long-term goal from global policy
                g_value, g_action, g_action_log_prob, g_rec_states = \
                    g_policy.act(
                        g_rollouts.obs[g_step + 1],
                        g_rollouts.rec_states[g_step + 1],
                        g_rollouts.masks[g_step + 1],
                        extras=g_rollouts.extras[g_step + 1],
                        deterministic=False
                    )
                cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
                global_goals = [[int(action[0] * local_w),
                                int(action[1] * local_h)]
                                for action in cpu_actions]

                g_reward = 0
                g_masks = torch.ones(num_scenes).float().to(device)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Get short term goal
            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = planner_pose_inputs[e]
                p_input['goal'] = global_goals[e]

            output = envs.get_short_term_goal(planner_inputs)
            # ------------------------------------------------------------------

            ### TRAINING
            torch.set_grad_enabled(True)
       

            # ------------------------------------------------------------------
            # Train Local Policy
            if (l_step + 1) % args.local_policy_update_freq == 0 \
                    and args.train_local:
                local_optimizer.zero_grad()
                policy_loss.backward()
                local_optimizer.step()
                l_action_losses.append(policy_loss.item())
                policy_loss = 0
                local_rec_states = local_rec_states.detach_()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Train Global Policy

            # if g_step % args.num_global_steps == 10 and l_step == 24: 
            if g_step % args.num_global_steps == args.num_global_steps - 1 \
                    and l_step == args.num_local_steps - 1:

                if args.train_global:
                    print(" ============================ train global ")
                    g_next_value = g_policy.get_value(
                        g_rollouts.obs[-1],
                        g_rollouts.rec_states[-1],
                        g_rollouts.masks[-1],
                        extras=g_rollouts.extras[-1]
                    ).detach()

                    g_rollouts.compute_returns(g_next_value, args.use_gae,
                                            args.gamma, args.tau)
                    g_value_loss, g_action_loss, g_dist_entropy = \
                        g_agent.update(g_rollouts)
                    g_value_losses.append(g_value_loss)
                    g_action_losses.append(g_action_loss)
                    g_dist_entropies.append(g_dist_entropy)
                g_rollouts.after_update()
            # ------------------------------------------------------------------

            # Finish Training
            torch.set_grad_enabled(False)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Logging
            if total_num_steps % args.log_interval == 0:
                end = time.time()
                time_elapsed = time.gmtime(end - start)
                log = " ".join([
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(total_num_steps *
                                            num_scenes),
                    "FPS {},".format(int(total_num_steps * num_scenes \
                                        / (end - start)))
                ])

                log += "\n\tRewards:"

                if len(g_episode_rewards) > 0:
                    log += " ".join([
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards),
                            np.median(per_step_g_rewards)),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards))
                    ])
                writer.add_scalar('Global/ep_mean_rew', np.mean(g_episode_rewards), total_num_steps)
                writer.add_scalar('Global/step_mean_rew', np.mean(per_step_g_rewards), total_num_steps)
                writer.add_scalar('Global/loss', np.mean(g_value_losses), total_num_steps)

                log += "\n\tLosses:"

                if args.train_local and len(l_action_losses) > 0:
                    log += " ".join([
                        " Local Loss:",
                        "{:.3f},".format(
                            np.mean(l_action_losses))
                    ])

                if args.train_global and len(g_value_losses) > 0:
                    log += " ".join([
                        " Global Loss value/action/dist:",
                        "{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies))
                    ])

                if args.train_slam and len(costs) > 0:
                    log += " ".join([
                        " SLAM Loss proj/exp/pose:"
                        "{:.4f}/{:.4f}/{:.4f}".format(
                            np.mean(costs),
                            np.mean(exp_costs),
                            np.mean(pose_costs))
                    ])

                print(log)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Save best models
            if (total_num_steps * num_scenes) % args.save_interval < \
                    num_scenes:

                # Save Neural SLAM Model
                if len(costs) >= 1000 and np.mean(costs) < best_cost \
                        and not args.eval:
                    best_cost = np.mean(costs)
                    torch.save(nslam_module.state_dict(),
                            os.path.join(log_dir, "model_best.slam"))

                # Save Local Policy Model
                if len(l_action_losses) >= 100 and \
                        (np.mean(l_action_losses) <= best_local_loss) \
                        and not args.eval:
                    torch.save(l_policy.state_dict(),
                            os.path.join(log_dir, "model_best.local"))

                    best_local_loss = np.mean(l_action_losses)

                # Save Global Policy Model
                if len(g_episode_rewards) >= 50 and \
                        (np.mean(g_episode_rewards) >= best_g_reward) \
                        and not args.eval:
                    torch.save(g_policy.state_dict(),
                            os.path.join(log_dir, "model_best.global"))
                    best_g_reward = np.mean(g_episode_rewards)

            # Save periodic models
            if (total_num_steps * num_scenes) % args.save_periodic < \
                    num_scenes:
                step = total_num_steps * num_scenes
                if args.train_slam:
                    torch.save(nslam_module.state_dict(),
                            os.path.join(dump_dir,
                                            "periodic_{}.slam".format(step)))
                if args.train_local:
                    torch.save(l_policy.state_dict(),
                            os.path.join(dump_dir,
                                            "periodic_{}.local".format(step)))
                if args.train_global:
                    torch.save(g_policy.state_dict(),
                            os.path.join(dump_dir,
                                            "periodic_{}.global".format(step)))
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
