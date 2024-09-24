import numpy as np 
import skimage
import skimage.morphology
import cv2 
import matplotlib.pyplot as plt 

import torch
import skfmm
from scipy.optimize import linear_sum_assignment
from random import shuffle
from scipy.ndimage import label


def kmeans(np_obstacle_map, np_frontier_map, k):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    # geodesic distance

    is_frontier = np_frontier_map == 1
    frontier_idx = np.where(is_frontier)
    rows = frontier_idx[0].shape[0]
    if rows == 0:
        return None, None, None, None
    if rows < k:
        k = rows

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    # the Forgy method will fail if the whole array contains the same rows
    cluster_idx = np.arange(rows)
    shuffle(cluster_idx)
    cluster_idx = cluster_idx[:k]

    for count in range(5):

        for k_i, ci in enumerate(cluster_idx):
            cx, cy = frontier_idx[0][ci], frontier_idx[1][ci]

            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
            np_obstacle_map_frontierK[cx, cy] = 1
            np_obstacle_map_distance = skfmm.distance(1 - np_obstacle_map_frontierK)
            distances[:, k_i] = np_obstacle_map_distance[frontier_idx[0], frontier_idx[1]]

            # distances[:, k_i] = (frontier_idx[0] - cx)**2 + (frontier_idx[1] - cy)**2
            
            
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for k_i in range(k):
            nearest_clusters_k_i_idx = np.where(nearest_clusters == k_i)
            if not nearest_clusters_k_i_idx:
                continue
            
            nearest_clusters_k_i_idx = nearest_clusters_k_i_idx[0]

            if frontier_idx[0][nearest_clusters_k_i_idx].size == 0 or frontier_idx[1][nearest_clusters_k_i_idx].size == 0:
                continue

            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
            np_obstacle_map_frontierK[int(frontier_idx[0][nearest_clusters_k_i_idx].mean()), int(frontier_idx[1][nearest_clusters_k_i_idx].mean())] = 1
            np_obstacle_map_distance = skfmm.distance(1 - np_obstacle_map_frontierK)

            # cx = int(frontier_idx[0][nearest_clusters_k_i_idx].mean())
            # cy = int(frontier_idx[1][nearest_clusters_k_i_idx].mean())

            # np_obstacle_map_distance = distance_map(np_obstacle_map.shape, cx, cy)
            
            
            temp_distance_k_i = np_obstacle_map_distance[is_frontier][nearest_clusters_k_i_idx]
            cluster_idx[k_i] = nearest_clusters_k_i_idx[np.argmin(temp_distance_k_i)]

        last_clusters = nearest_clusters

    return (frontier_idx[0][cluster_idx], frontier_idx[1][cluster_idx]), nearest_clusters, is_frontier, frontier_idx


def co_scan_selection(args, full_map, full_pose):
    combined_full_map = torch.sum(full_map.detach().clone(), dim=0)
    explored_map = combined_full_map[1].detach().clone().cpu().numpy()
    explored_region = cv2.erode(cv2.dilate(explored_map, np.ones((3,3))), np.ones((3,3)))

    obstacle_boundary = 20 
    unit_size_cm = 10 
    selem = skimage.morphology.disk(obstacle_boundary / unit_size_cm)
    obstacle_map = combined_full_map[0].detach().clone().cpu().numpy()
    binary_obstacle_map = (obstacle_map > 0.5).astype(np.uint8)
    obstacle_region = skimage.morphology.binary_dilation(binary_obstacle_map, selem)
    explored_boundary = explored_region - cv2.erode(explored_region, np.ones((3,3)))
    frontier = np.clip(explored_boundary - obstacle_region, 0.0, 1.0) 

    np_obstacle_map = (binary_obstacle_map > 0.5) # 1120, 1120 
    np_frontier_map = (frontier > 0.5).astype(np.float32)

    labeled_image, num_labels = label(np_frontier_map)
    component_sizes = np.bincount(labeled_image.flatten())
    mask = component_sizes[labeled_image] >= 3
    filtered_image = np_frontier_map.copy()
    filtered_image[~mask] = 0
    np_frontier_map = filtered_image.copy()

    # agent_pos = full_pose[optimized_env,:2].cpu().numpy().astype(np.int) 
    locs = full_pose[:,:2].cpu().numpy()

    agent_pos = np.array([[int(r * 100.0/args.map_resolution), int(c * 100.0/args.map_resolution)] for c, r in locs])

    num_agent = agent_pos.shape[0]
    np_obstacle_map_distance = []
    dd_mask = np.ones(np_obstacle_map.shape, dtype=np.bool)
    for k_i in range(num_agent):
        np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
        np_obstacle_map_frontierK[agent_pos[k_i, 0], agent_pos[k_i, 1]] = 1
        dd = skfmm.distance(1 - np_obstacle_map_frontierK)
        np_obstacle_map_distance.append(dd)
        if type(dd) is np.ndarray:
            dd_mask[:] = False
        else:
            dd_mask &= dd.mask
    np_frontier_map[dd_mask] = 0
    cluster_center, nearest_clusters, is_frontier, frontier_idx = kmeans(np_obstacle_map, np_frontier_map, num_agent)
    # nearest_clusters: (n_frontier,)
    if cluster_center is None:
        global_goals = [None] * num_agent
    else:
        nc = cluster_center[0].shape[0]
        cost = np.zeros((num_agent, nc))
        # n_agent x n_cluster
        for k_i in range(num_agent):
            cost[k_i, :] = np_obstacle_map_distance[k_i][cluster_center[0], cluster_center[1]]

        cost = np.hstack([cost] * ((num_agent - 1) // nc + 1))

        global_goals = np.zeros((num_agent, 2), dtype=np.int32)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(num_agent):
            cluster_idx = col_ind[i] % nc
            agent_idx = row_ind[i]
            frontier_dist = np_obstacle_map_distance[agent_idx][is_frontier]
            frontier_dist[nearest_clusters != cluster_idx] = np.inf
            select = np.argmin(frontier_dist) 
            global_goals[agent_idx] = [frontier_idx[0][select], frontier_idx[1][select]]

    return np_obstacle_map, np_frontier_map, global_goals

def get_coscan_goal(args, prev_goals, temp_global_goals, full_pose, full_map):
    map_combined_envs = [1]*args.num_processes
    np_obstacle_map, np_frontier_map, optimized_env_global_goals = co_scan_selection(args, full_map, full_pose)

    print("opt goal: ",optimized_env_global_goals)
    global_goals = []
    for i in range(full_pose.shape[0]):
        if None in optimized_env_global_goals: 
            global_goals.append(temp_global_goals[i])
        else: 
            opt_goal = optimized_env_global_goals[i].tolist()
            if opt_goal not in prev_goals[i]:
                goal = optimized_env_global_goals[i].tolist()
                global_goals.append([goal[1], goal[0]])
            else : 
                global_goals.append(temp_global_goals[i])

    # plt.imsave("temp0f.png", np_frontier_map)
    # plt.imsave("temp0o.png", np_obstacle_map)

    print("global goal: ",optimized_env_global_goals)

    return global_goals 
