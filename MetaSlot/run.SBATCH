#!/bin/bash

#SBATCH --output=/home/yy2694/scratch/SlotAttn/slurm_outputs/%x_%j.out
#SBATCH --error=/home/yy2694/scratch/SlotAttn/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wz3008@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=7-metaslot

module purge
singularity exec --nv \
    --overlay /scratch/yy2694/data/saycam/5fps.sqf:ro \
    --overlay /scratch/yy2694/data/saycam/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
    --overlay /home/yy2694/scratch/overlay-50G-10M-mateslot.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh;conda activate MetaSlot; cd /home/yy2694/scratch/SlotAttn/MetaSlot; \
    python train.py \
        --data_dir  / \
        --cfg_file /home/yy2694/scratch/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
        --ckpt_file /home/yy2694/scratch/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
        --save_dir ./save \
        --use_dp"

# python -m pdb train.py \
#         --data_dir  / \
#         --cfg_file ./Config/config-metaslot/dinosaur-saycam-ViT_B16.py \
#         --save_dir ./save \
#         --ckpt_file dino_s_vitb16