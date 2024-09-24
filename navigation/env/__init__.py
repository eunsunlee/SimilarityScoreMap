import torch

from .habitat import construct_envs, construct_envs_largescene, construct_envs_largescene_hybrid

def make_vec_envs_largescene_hybrid(args, ep_num, add_ind):
    envs = construct_envs_largescene_hybrid(args, ep_num, add_ind)
    envs = VecPyTorch(envs, args.device)
    return envs

def make_vec_envs_largescene(args, ep_num, add_ind):
    envs = construct_envs_largescene(args, ep_num, add_ind)
    envs = VecPyTorch(envs, args.device)
    return envs

def make_vec_envs(args, ep_num):
    envs = construct_envs(args, ep_num)
    envs = VecPyTorch(envs, args.device)
    return envs


# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.device = device

    def reset(self):
        obs, depth, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, depth, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, depth, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, depth, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, depth, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, depth, reward, done, info

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def get_short_term_goal(self, inputs):
        stg = self.venv.get_short_term_goal(inputs)
        stg = torch.from_numpy(stg).float()
        return stg

    def close(self):
        return self.venv.close()
