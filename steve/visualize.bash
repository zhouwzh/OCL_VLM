module purge
singularity exec --nv \
    --overlay /scratch/yy2694/data/saycam/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
    --overlay /home/yy2694/scratch/overlay-50G-10M-slotattn.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \

source /ext3/env.sh;
conda activate steve;
cd /scratch/yy2694/SlotAttn/steve;

python visualize.py \
    --num_slots 6 \
    --batch_size 1 \
    --checkpoint_path \
    /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-21T03:07:22.328408_s_5fps_slot_num_6/best_model_at_epoch_60.pt \
    --use_dvae \
    --vocab_size 4096 \
    --dataset saycam \
    --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
    --data_path '/saycam_transcript_5fps/' \
    --start_epoch 61

python visualize.py \
    --num_slots 8 \
    --batch_size 1 \
    --checkpoint_path \
    /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-21T03:07:34.258846_s_5fps_slot_num_8/best_model.pt \
    --use_dvae \
    --vocab_size 4096 \
    --dataset saycam \
    --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
    --data_path '/saycam_transcript_5fps/' \
    --start_epoch 61

python visualize.py \
    --num_slots 11 \
    --batch_size 1 \
    --checkpoint_path \
    /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-21T03:07:34.837357_s_5fps_slot_num_11/best_model.pt \
    --use_dvae \
    --vocab_size 4096 \
    --dataset saycam \
    --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
    --data_path '/saycam_transcript_5fps/' \
    --start_epoch 61

# python visualize.py \
#     --num_slots 11 \
#     --batch_size 1 \
#     --checkpoint_path \
#     /scratch/yy2694/SlotAttn/steve/logs/2025-10-18T16:29:40.831746_s_5fps_slot_num_11/best_model_at_epoch_30.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 31

# python visualize.py \
#     --ep_len 1 \
#     --batch_size 1 \
#     --checkpoint_path \
#     /scratch/yy2694/SlotAttn/steve/logs/2025-10-17T02:01:00.207951_s_5fps_eplen_1/best_model_until_200000_steps.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 13

# python visualize.py \
#     --batch_size 1 \
#     --checkpoint_path \
#     /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-13T16:38:40.712423_s_5fps/best_model_at_epoch_50.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 50

# python visualize.py \
#     --batch_size 1 \
#     --checkpoint_path \
#     /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-13T16:38:40.712423_s_5fps/best_model_at_epoch_60.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 60

# python visualize.py \
#     --batch_size 1 \
#     --checkpoint_path \
#     /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-14T09:57:34.099631_s_5fps/best_model_at_epoch_70.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 70

# python visualize.py \
#     --batch_size 1 \
#     --checkpoint_path \
#     /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-14T09:57:34.099631_s_5fps/best_model_at_epoch_80.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 80


############################################
# python visualize_ttt.py \
#     --batch_size 1 \
#     --checkpoint_path \
#     /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-12T21:11:45.777061_s_5fps_epoch_45/best_model_at_epoch_30.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 30

# python visualize_ttt.py \
#     --batch_size 1 \
#     --checkpoint_path \
#     /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-12T21:11:45.777061_s_5fps_epoch_45/best_model_at_epoch_40.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 40

# python visualize_ttt.py \
#     --batch_size 1 \
#     --checkpoint_path \
#     /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-13T16:38:40.712423_s_5fps/best_model_at_epoch_50.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 50

# python visualize_ttt.py \
#     --batch_size 1 \
#     --checkpoint_path \
#     /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-13T16:38:40.712423_s_5fps/best_model_at_epoch_60.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 60

# python visualize_ttt.py \
#     --batch_size 1 \
#     --checkpoint_path \
#     /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-14T09:57:34.099631_s_5fps/best_model_at_epoch_70.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 70

# python visualize_ttt.py \
#     --batch_size 1 \
#     --checkpoint_path \
#     /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-14T09:57:34.099631_s_5fps/best_model_at_epoch_80.pt \
#     --use_dvae \
#     --vocab_size 4096 \
#     --dataset saycam \
#     --json_path '/scratch/yy2694/data/saycam/saycam_transcript_frames/' \
#     --data_path '/saycam_transcript_5fps/' \
#     --start_epoch 80
