import numpy as np 
import gtsam 
from itertools import combinations
from utils.landmarks import Landmarks
import torch 
from utils.mapper import Remap_Module
import time
from utils.graph import GraphMap, Graph_Mapper
import env.habitat.utils.pose as pu
import matplotlib.pyplot as plt


def multi_agent_optimization(args, num_envs, pose_graphs, full_map, full_pose, device):

    map_combined = 0 
    map_size = args.map_size_cm // args.map_resolution

    landmark_search = Landmarks(args)
    re_mapper = Remap_Module(args).to(device)

    graph =  gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.3, 0.3, 0.1))
    MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001, 0.0001]))

    st = time.time()

    for i in range(num_envs): 
        if i == 0 : 
            PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.9, 0.9, 0.8))
        else: 
            PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(30,30,10))

        for j in range(len(pose_graphs[i].nodes)):

            env_id = str(i)
            pose_x, pose_y, pose_th = pose_graphs[i].node_at(j).pose[0].cpu().numpy()
            initial_estimate.insert(gtsam.symbol(env_id, j), gtsam.Pose2(pose_x, pose_y, pose_th*np.pi/180.))
            if j == 0: 
                graph.add(gtsam.PriorFactorPose2(gtsam.symbol(env_id, j), gtsam.Pose2(pose_x, pose_y, pose_th), PRIOR_NOISE))
            else:
                graph.add(gtsam.BetweenFactorPose2(gtsam.symbol(env_id, j-1), gtsam.symbol(env_id, j), \
                            gtsam.Pose2(pose_graphs[i].edge_at(j-1, j).sensor_pose), ODOMETRY_NOISE))



    no_landmark = 0 
    

    rep_feats = []
    for i in range(num_envs): 
        rep_feats.append(pose_graphs[i].get_rep_node_feat())


    unique_env_pairs = list(combinations(list(range(num_envs)), 2))
    for (env0, env1) in unique_env_pairs: 
        for (cur_rep_node, cur_feat) in zip(pose_graphs[env0].rep_nodes, rep_feats[env0]): 
            logits = np.matmul(cur_feat, rep_feats[env1].transpose(0,2,1))
            top_sim_idx = np.argsort(logits.flatten())[::-1][0]
            top_sim_node =  list(pose_graphs[env1].rep_nodes)[top_sim_idx]

            landmark_added, landmark_attr = landmark_search.find_landmark(pose_graphs[env0].node_at(cur_rep_node), pose_graphs[env1].node_at(top_sim_node))
            if landmark_added:
                no_landmark += 1
                initial_estimate.insert(gtsam.symbol('L', no_landmark), gtsam.Point2(landmark_attr[0], landmark_attr[1]))
                graph.push_back(gtsam.BearingRangeFactor2D(gtsam.symbol(str(env0),cur_rep_node), gtsam.symbol('L',no_landmark),  gtsam.gtsam.Rot2.fromDegrees(-landmark_attr[2]), landmark_attr[3], MEASUREMENT_NOISE))
                graph.push_back(gtsam.BearingRangeFactor2D(gtsam.symbol(str(env1),top_sim_node), gtsam.symbol('L',no_landmark),  gtsam.gtsam.Rot2.fromDegrees(-landmark_attr[4]), landmark_attr[5], MEASUREMENT_NOISE))

    if no_landmark > 3 : 
        try: 
            parameters = gtsam.LevenbergMarquardtParams()
            parameters.setRelativeErrorTol(1e-2)
            parameters.setMaxIterations(1000)
    
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, parameters)
            result = optimizer.optimize()

            marginals = gtsam.Marginals(graph, result)

            new_full_map = torch.zeros(num_envs, 4, map_size, map_size).float().to(device)
            new_full_pose = torch.zeros(num_envs, 3).float().to(device)

            for i in range(num_envs): 

                for n in range(1,len(pose_graphs[i].nodes)):
                    r_pose = result.atPose2(gtsam.symbol(str(i),n))
                    t_pose = torch.tensor([r_pose.x(), r_pose.y(), r_pose.theta()*180/np.pi]).unsqueeze(0).to(device)

                    new_full_pose[i,0] = (t_pose[0,0].detach().clone() * 100 / args.map_resolution) 
                    new_full_pose[i,1] = (t_pose[0,1].detach().clone() * 100 / args.map_resolution) 
                    new_full_pose[i,2] = (t_pose[0,2].detach().clone())
                    locs = new_full_pose[i].cpu().numpy()
                    r, c = int(locs[1]), int(locs[0])
                    new_full_map[i, 2:, r - 1:r + 2, c - 1:c + 2] = 1.

                    proj_map = pose_graphs[i].node_at(n).proj_map.to(device)
                    exp_map = pose_graphs[i].node_at(n).exp_map.to(device)
                    new_full_map[i:i+1,0,:,:], new_full_map[i:i+1,1,:,:] = re_mapper(proj_map, exp_map, t_pose, new_full_map[i:i+1,0,:,:], new_full_map[i:i+1,1,:,:])

            max_values, _ = torch.max(new_full_map, dim=0)
            combined_map = max_values.unsqueeze(0)
            map_combined = 1
            end_time = time.time()

            new_pose_graphs = []
            for i in range(num_envs): 
                new_pose_graph = GraphMap(device)
                new_pose_graph.rep_nodes = pose_graphs[i].rep_nodes
                
                rgb = pose_graphs[i].node_at(0).rgb
                depth = pose_graphs[i].node_at(0).depth
                pose = pose_graphs[i].node_at(0).pose
                proj_map = pose_graphs[i].node_at(0).proj_map
                exp_map = pose_graphs[i].node_at(0).exp_map
                img_feat = pose_graphs[i].node_at(0).img_feat

                if i == 0 : 
                    new_pose = np.array(pose[0])
                if i > 0 : # calculate initial pose relative to the first robot
                    env0_pose = np.array(pose_graphs[0].node_at(0).pose[0])
                    env_0_r_pose = result.atPose2(gtsam.symbol(str(0),0))
                    env_c_r_pose = result.atPose2(gtsam.symbol(str(i),0))

                    np_t_pose_0 = np.array([env_0_r_pose.x(), env_0_r_pose.y(), env_0_r_pose.theta()])
                    np_t_pose_c = np.array([env_c_r_pose.x(), env_c_r_pose.y(), env_c_r_pose.theta()])

                    dx, dy, do = pu.get_rel_pose_change(np_t_pose_c, np_t_pose_0)
                    new_pose = pu.get_new_pose(env0_pose ,(dx, dy, do))


                new_pose_graph.add_robot_node(rgb, depth, new_pose, proj_map, exp_map,  img_feat) 



                for n in range(1,len(pose_graphs[i].nodes)): 

                    r_pose_0 = result.atPose2(gtsam.symbol(str(i),n-1))
                    r_pose_1 = result.atPose2(gtsam.symbol(str(i),n))

                    np_t_pose_0_ = np.array([r_pose_0.x(), r_pose_0.y(), r_pose_0.theta()])
                    np_t_pose_1_ = np.array([r_pose_1.x(), r_pose_1.y(), r_pose_1.theta()])

                    dx, dy, do = pu.get_rel_pose_change(np_t_pose_1_, np_t_pose_0_)

                    sensor_pose = (dx,dy,(do))
                    gt_sensor_pose = pose_graphs[i].edge_at(n-1,n).gt_sensor_pose
                    new_pose = pu.get_new_pose(new_pose ,(dx, dy, (do)))
                    rgb = pose_graphs[i].node_at(n).rgb
                    depth = pose_graphs[i].node_at(n).depth
                    pose = pose_graphs[i].node_at(n).pose
                    proj_map = pose_graphs[i].node_at(n).proj_map
                    exp_map = pose_graphs[i].node_at(n).exp_map
                    img_feat = pose_graphs[i].node_at(n).img_feat

                    # 1,2,3,4,.... 
                    new_pose_graph.add_robot_node(rgb, depth, new_pose, proj_map, exp_map,  img_feat) 

                    # 0-1, 1-2, .... 
                    new_pose_graph.add_robot_edge(new_pose_graph.node_at(n-1), \
                        new_pose_graph.node_at(n), sensor_pose, gt_sensor_pose, 0) 
                new_pose_graphs.append(new_pose_graph)

            tmp_map_module = Graph_Mapper(args).to(device)
            remapped_full_map = torch.zeros(num_envs, 4, map_size, map_size).float().to(device)

            for e in range(num_envs):

                tmp_full_pose = torch.zeros(1, 3).float().to(device)
                tmp_full_map = torch.zeros(1, 4, map_size, map_size).float().to(device)
                freq_map = torch.zeros(1, 4, map_size, map_size).float().to(device)
                tmp_full_pose[0,:] =  torch.tensor(new_pose_graphs[e].node_at(0).pose).to(device)

                for i in range(1,len(new_pose_graphs[e].nodes)):
                    tmp_pose = torch.tensor(new_pose_graphs[e].edge_at(i-1, i).sensor_pose).unsqueeze(0).to(device)
                    tmp_proj = new_pose_graphs[e].node_at(i).proj_map.to(device)
                    tmp_exp = new_pose_graphs[e].node_at(i).exp_map.to(device)


                    tmp_full_map[:, 0, :, :], tmp_full_map[:, 1, :, :], tmp_full_pose, freq_map = \
                        tmp_map_module(tmp_proj, tmp_exp, tmp_pose, tmp_full_map[:, 0, :, :],
                            tmp_full_map[:, 1, :, :], tmp_full_pose, 1, freq_map)  
                remapped_full_map[e:e+1] = tmp_full_map


            return map_combined, combined_map, new_full_pose, new_pose_graphs, remapped_full_map

        except: 
            print("error")
            map_combined = 0
            return map_combined, None, None, None, None
    return map_combined, None, None, None, None