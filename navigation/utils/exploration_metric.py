from torch.nn import functional as F

import torch 

def get_exp_ratio(reset_observations, infos, prev_explored_area):
    # get explorable map 
    explorable_map0 = reset_observations[0]['explorable_map']
    explorable_map1 = reset_observations[1]['explorable_map']
    # episode map information 
    episode_info0 = reset_observations[0]['temp_info']
    episode_info1 = reset_observations[1]['temp_info']

    rotated0 = episode_info0['rotated']
    rotated1 = episode_info1['rotated']
    boundary0 = episode_info0['boundary']
    boundary1 = episode_info1['boundary']


    # get explored0 
    explored_0 = infos[0]['explored_map']
    # get explored1 
    explored_1 = infos[1]['explored_map']
    r_rotated1 = torch.zeros_like(rotated1)

    if boundary0[0] == 1 and boundary1[0] == 1: 
        r_rotated1[0,0, boundary1[1]:boundary1[2], boundary1[3]:boundary1[4]] = torch.tensor(explored_1)
        r_translated1_exp = F.grid_sample(r_rotated1, episode_info1['rot_mat_inv'])
        r_gridmap1_exp = F.grid_sample(r_translated1_exp, episode_info1['trans_mat_inv'])
        r_translated0_exp = F.grid_sample(r_gridmap1_exp, episode_info0['trans_mat'])
        r_rotated0_exp = F.grid_sample(r_translated0_exp, episode_info0['rot_mat'])
        r_explorable_0 = r_rotated0_exp[0,0, boundary0[1]:boundary0[2], boundary0[3]:boundary0[4]]  
    else: 
        r_rotated1[0,0,:] = torch.tensor(explored_1)[boundary1[1]:boundary1[2], boundary1[3]:boundary1[4]]
        r_translated1_exp = F.grid_sample(r_rotated1, episode_info1['rot_mat_inv'])
        r_gridmap1_exp = F.grid_sample(r_translated1_exp, episode_info1['trans_mat_inv'])
        r_translated0_exp = F.grid_sample(r_gridmap1_exp, episode_info0['trans_mat'])
        r_rotated0_exp = F.grid_sample(r_translated0_exp, episode_info0['rot_mat'])

        r_explorable_0 = torch.zeros_like(torch.tensor(explored_0))
        r_explorable_0[boundary0[1]:boundary0[2], boundary0[3]:boundary0[4] ]  = r_rotated0_exp[0,0]

    new_explored_1 = r_explorable_0.detach().cpu().numpy()

    new_explored_1[new_explored_1 >= 0.5] = 1
    new_explored_1[new_explored_1 < 0.5] = 0 

    explored_0[explorable_map0 == 0] = 0 
    new_explored_1[explorable_map0 == 0] = 0 

    explored_sum = explored_0 + new_explored_1
    explored_sum[explored_sum == 2] = 1.

    curr_explored = explorable_map0 * explored_sum #
    curr_explored_area = curr_explored.sum()

    # curr_explored_area = explored_sum.sum()
    reward_scale = explorable_map0.sum()
    m_reward = (curr_explored_area - prev_explored_area) * 1. 
    m_ratio = m_reward/reward_scale

    m_reward = m_reward * 25./10000. # converting to m^2
    m_reward *= 0.02 # Reward Scaling

    return m_ratio, m_reward, curr_explored_area

