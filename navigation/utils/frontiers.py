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

def get_frontiers_multi(args, ref_map, freq_map, frontier_filter): 
    num_envs = ref_map.shape[0] 
    obstacle_boundary = 20 
    unit_size_cm = 10 
    frontier_list = []
    frontier_stats = []
    filtered_avg_freq_vals_list = []
    frontier_img = np.zeros_like(ref_map[:,0,:,:].cpu().numpy())

    explored_region_list = []
    for i in range(num_envs): 
        explored_map = ref_map[i,1].detach().clone().cpu().numpy()
        explored_region = cv2.erode(cv2.dilate(explored_map, np.ones((3,3))), np.ones((3,3)))
        explored_region_list.append(explored_region) 


    for i in range(num_envs): 

        selem = skimage.morphology.disk(obstacle_boundary / unit_size_cm)
        obstacle_map = ref_map[i,0].detach().clone().cpu().numpy()
        binary_obstacle_map = (obstacle_map > 0.5).astype(np.uint8)
        obstacle_region = skimage.morphology.binary_dilation(binary_obstacle_map, selem)
        explored_map = ref_map[i,1].detach().clone().cpu().numpy()
        # remove the small holes in the explored region
        explored_region = cv2.erode(cv2.dilate(explored_map, np.ones((3,3))), np.ones((3,3)))

        explored_boundary = explored_region - cv2.erode(explored_region, np.ones((3,3)))
        frontier = np.clip(explored_boundary - obstacle_region, 0.0, 1.0) 
        frontier_img[i] = frontier
        binary_frontier = (frontier > 0.5).astype(np.uint8)

        if frontier_filter: 
            filtered_binary_frontier = binary_frontier
            for e in range(num_envs): 
                if i != e: 
                    filtered_binary_frontier[explored_region_list[e] > 0] = 0

            num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(filtered_binary_frontier)
        else: 
            num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(binary_frontier)

        avg_freq_vals = []
        for f in range(num_labels): 
            if f != 0: 
                cur_label_binary_region = np.zeros_like(labeled_image)
                cur_label_binary_region[labeled_image == f] = 1.

                frontier_selem = skimage.morphology.disk(4)
                cur_frontier_region = skimage.morphology.binary_dilation(cur_label_binary_region, frontier_selem)
                frontier_w_freq = freq_map[i][0,1].cpu().numpy()
                frontier_w_freq[cur_frontier_region == 0] = 0 
                avg_freq_val = frontier_w_freq.sum() /len(np.where(frontier_w_freq > 0)[0])
                avg_freq_vals.append(avg_freq_val)

        # exclude the largest/first frontier = background 
        stats = stats[1:]
        centroids = centroids[1:]

        # excluding frontiers where the size is less than 5 
        frontier_size_values = stats[:,4]
        indices_to_keep = np.where(frontier_size_values > 4)
        filtered_centroids = centroids[indices_to_keep[0]]
        filtered_stats = stats[indices_to_keep[0]]
        filtered_avg_freq_vals = np.array(avg_freq_vals)[indices_to_keep[0]]

        frontier_list.append(filtered_centroids.astype(int))
        frontier_stats.append(filtered_stats)
        filtered_avg_freq_vals_list.append(filtered_avg_freq_vals)


    return frontier_list,frontier_stats, obstacle_map, obstacle_region, explored_map, frontier_img, filtered_avg_freq_vals_list 


def get_frontier_based_global_goals_multi( args, temp_global_goals, frontiers, stats, full_pose, map_combined, filtered_avg_freq_vals_list, least_sim_node_poses): 

    global_goals = []
    selected_frontiers  = [] 
    for i in range(full_pose.shape[0]):
        if frontiers[i].shape[0] == 0 : 
            global_goals.append(temp_global_goals[i])
        else: 
            if map_combined and len(selected_frontiers) > 0 : 
                # removing pre-selected frontier 
                selected_set = set(map(tuple, selected_frontiers))
                frontiers_set = set(map(tuple, frontiers[i]))
                frontiers[i] = np.array(list(frontiers_set - selected_set))
                if frontiers[i].shape[0] == 0 : 
                    print("frontier  is 0 (inside function)")
                    global_goals = [[141,165], [165,185]]
                    return global_goals

            frontier_to_cur_pose_dist = np.linalg.norm(frontiers[i] - full_pose[i].detach().cpu().numpy()[:2], axis=1)
            # TODO: change to largest frontier ? 
            if args.frontier_type == "closest": 
                closest_frontier = frontiers[i][np.argmin(frontier_to_cur_pose_dist)]
                global_goals.append([closest_frontier[1], closest_frontier[0]])
                selected_frontiers.append([closest_frontier[0], closest_frontier[1]])  

            elif args.frontier_type == "farthest":
                farthest_frontier = frontiers[i][np.argmax(frontier_to_cur_pose_dist)]
                global_goals.append([farthest_frontier[1], farthest_frontier[0]])
                selected_frontiers.append([farthest_frontier[0], farthest_frontier[1]])  

            elif args.frontier_type == "largest": 
                largest_frontier_index = np.argmax(stats[i][:,4])
                largest_frontier = frontiers[i][largest_frontier_index]
                global_goals.append([largest_frontier[1], largest_frontier[0]])
                selected_frontiers.append([largest_frontier[0], largest_frontier[1]])   

            elif args.frontier_type == "least_freq": # based on combined map 
                least_freq_frontier_index = np.argmin(filtered_avg_freq_vals_list[i])
                least_freq_frontier = frontiers[i][least_freq_frontier_index]
                global_goals.append([least_freq_frontier[1], least_freq_frontier[0]])
                selected_frontiers.append([least_freq_frontier[0], least_freq_frontier[1]])   

            elif args.frontier_type == "least_sim_node": 
                frontier_to_least_sim_node = np.linalg.norm(frontiers[i] - least_sim_node_poses[i].detach().cpu().numpy()[:2], axis=1)
                closest_frontier = frontiers[i][np.argmin(frontier_to_least_sim_node)]
                global_goals.append([closest_frontier[1], closest_frontier[0]])
                selected_frontiers.append([closest_frontier[0], closest_frontier[1]])  

    return global_goals 
