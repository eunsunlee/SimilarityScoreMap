# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api

import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from .exploration_env import Exploration_Env
from .habitat_api.habitat.core.vector_env import VectorEnv
from .habitat_api.habitat_baselines.config.default import get_config as cfg_baseline


def make_env_fn(args, config_env, config_baseline, rank):
    dataset = PointNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    print("Loading {}".format(config_env.SIMULATOR.SCENE))
    config_env.freeze()

    env = Exploration_Env(args=args, rank=rank,
                          config_env=config_env, config_baseline=config_baseline, dataset=dataset
                          )

    env.seed(rank)
    return env


def construct_envs(args, ep_num):
    env_configs = []
    baseline_configs = []
    args_list = []

    basic_config = cfg_env(config_paths=
                           ["env/habitat/habitat_api/configs/" + args.task_config])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)
    # scenes = ['1pXnuDYAj8r','2n8kARJN3HM','759xd9YjKW5','JeFG25nYj2p','JmbYfDe2QKZ','VzqfbhrpDEA','gTV8FGcVJC9','uNb9QFRL6hY','vyrNrziPKCB']
    # 8WUmhLawc2A jtcxE69GiFV pa4otMbVnkk
    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    config_env = cfg_env(config_paths=
                            ["env/habitat/habitat_api/configs/" + args.task_config])
    config_env.defrost()

    if len(scenes) > 0:
        # config_env.DATASET.CONTENT_SCENES = scenes[
        #                                     i * scene_split_size: (i + 1) * scene_split_size
        #                                     ]

        config_env.DATASET.CONTENT_SCENES = scenes[ep_num:ep_num+1]

    gpu_id = 0

    config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

    agent_sensors = []
    agent_sensors.append("RGB_SENSOR")
    agent_sensors.append("DEPTH_SENSOR")

    config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

    config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
    config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

    config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
    config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
    config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
    config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

    config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
    config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
    config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
    config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

    config_env.SIMULATOR.TURN_ANGLE = 10
    config_env.DATASET.SPLIT = args.split

    config_env.freeze()

    config_baseline = cfg_baseline()


    envs = make_env_fn(args, config_env, config_baseline, 0 )


    return envs