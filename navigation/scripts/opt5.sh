python main_largescene_opt.py \
    --load_local  "pretrained_models/model_best.local" \
    --load_slam "pretrained_models/model_best.slam" \
    --load_global "0" \
    --eval 1  --exp_name "5_opt4" \
    --split train --task_config tasks/pointnav_mp3d.yaml --max_episode_length 1001 \
    --global_downscaling 1 --map_size_cm 5600 --num_global_steps 20 \
    --noisy_actions 1 --noisy_odometry 0 --use_pose_estimation 0  --depth_est 0 \
    --num_episodes 61 --num_processes 5 --traj_per_scene 1 \
    --frontier_type "least_freq" --frontier_filter 0 --num_local_steps 50 \
    --dump_location "./tmp_opt4/" --add_ind 12