#!/bin/bash

#SBATCH --output=/scratch/wz3008/cvcl-related/slurm_outputs/%x_%j.out
#SBATCH --error=/scratch/wz3008/cvcl-related/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wz3008@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=clip_metaslot


module purge
singularity exec --nv \
    --overlay /scratch/wz3008/data/saycam/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
    --overlay /scratch/wz3008/overlay-50G-10M-slotattn-test.ext3:ro \
    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh;conda activate MetaSlot; cd /scratch/wz3008/SlotAttn/; \
    python train_clip.py \
        --gpus 4 \
        --data_config /scratch/wz3008/SlotAttn/clip/config/data_config.yaml \
        --model_config /scratch/wz3008/SlotAttn/clip/config/model_config.yaml \
        --train_config /scratch/wz3008/SlotAttn/clip/config/train_config.yaml \
        --arch metaslot \
        --exp_name after_mean \
        --mean_epoch 15 \
        --augment_frames \
        --ckpt_file /home/wz3008/scratch/SlotAttn/clip/saved_checkpoints/2025-10-28-01-59-53_nofreeze_ori_config/mean_best_ckpt.pth"

# module purge
# singularity exec --nv \
#     --overlay /scratch/wz3008/data/saycam/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
#     --overlay /scratch/wz3008/overlay-50G-10M-slotattn-test.ext3:ro \
#     /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
#     -c "source /ext3/env.sh;conda activate MetaSlot; cd /scratch/wz3008/SlotAttn/; \
#     python train_clip.py \
#         --gpus 4 \
#         --data_config /scratch/wz3008/SlotAttn/clip/config/data_config.yaml \
#         --model_config /scratch/wz3008/SlotAttn/clip/config/model_config.yaml \
#         --train_config /scratch/wz3008/SlotAttn/clip/config/train_config.yaml \
#         --arch metaslot \
#         --exp_name nofreeze_ori_config \
#         --mean_epoch 100 \
#         --augment_frames \
#         --ckpt_file /scratch/wz3008/SlotAttn/clip/saved_checkpoints/2025-10-27-23-53-16_freeze_ori_config/mean_checkpoint.pth"

# 2025-10-27-03-15-46_topk  0.2 warm 200epoch, not freeze, no logits
# 2025-10-27-22-26-38_topk  0.2 warm, not freeze, logits
# 2025-10-27-23-08-27_topk  0.1 warm, freeze, logits
# 2025-10-27-23-47-13_topk orig config, freeze

