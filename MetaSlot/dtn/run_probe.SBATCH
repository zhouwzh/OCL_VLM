#!/bin/bash

#SBATCH --output=/home/wz3008/scratch/SlotAttn/slurm_outputs/%x_%j.out
#SBATCH --error=/home/wz3008/scratch/SlotAttn/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wz3008@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --account torch_pr_36_mren
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=full_dataset_probe_eval

module purge
singularity exec --nv \
    --overlay /scratch/yy2694/data/saycam/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
    --overlay /scratch/wz3008/data/saycam_mask.sqf:ro \
    --overlay /scratch/wz3008/overlay-50G-10M-metaslot.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh;conda activate MetaSlot; cd /scratch/wz3008/SlotAttn/MetaSlot; \
    python train.py \
        --data_dir  /saycam_transcript_5fps/ \
        --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
        --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
        --save_dir ./save \
        --use_dp \
        --dev \
        --ttt_step 20 \
        --eval_fgari"

singularity exec --nv \
    --overlay /scratch/wz3008/overlay-50G-10M-metaslot.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh;conda activate MetaSlot; cd /scratch/wz3008/SlotAttn/MetaSlot; \
    python occupy_gpu.py"