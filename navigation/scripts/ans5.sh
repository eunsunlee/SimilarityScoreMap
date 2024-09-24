python main_largescene_ans.py \
    --load_local  "pretrained_models/model_best.local" \
    --load_slam "pretrained_models/model_best.slam" \
    --load_global "pretrained_models/model_best.global" \
    --eval 1  --exp_name "val_ans_5" \
    --split train --task_config tasks/pointnav_mp3d.yaml --max_episode_length 1001 \
    --global_downscaling 2 --map_size_cm 2400 --num_global_steps 20 \
    --noisy_actions 1 --noisy_odometry 0 --use_pose_estimation 0  --depth_est 1 \
    --num_episodes 61 --num_processes 5 --traj_per_scene 1 \
    --dump_location "./tmp_ans/"

