#!/bin/bash

#SBATCH --output=/home/yy2694/scratch/SlotAttn/slurm_outputs/%x_%j.out
#SBATCH --error=/home/yy2694/scratch/SlotAttn/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wz3008@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=11_steve

module purge
singularity exec --nv \
    --overlay /scratch/yy2694/data/saycam/5fps.sqf:ro \
    --overlay /home/yy2694/scratch/overlay-50G-10M-slotattn.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh; conda activate steve; cd /scratch/yy2694/SlotAttn/steve; \
    python train.py \
        --num_slots 11 \
        --data_path '/*' \
        --log_path /scratch/yy2694/SlotAttn/steve/logs/ \
        --use_dvae \
        --dataset samcam \
        --batch_size 96 \
        --exp_name s_5fps_slot_num_11 \
        --checkpoint_path /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-19T18:58:47.429698_s_5fps_slot_num_11/checkpoint.pt.tar"

# python -m tensorboard.main --logdir /home/yy2694/scratch/SlotAttn/steve/logs/2025-10-11T03:11:02.321018 --port 6006


# singularity exec --nv \
#     --overlay /scratch/yy2694/data/LifelongBenchTC.sqf:ro \
#     --overlay /home/yy2694/overlay-50G-10M-slotattn.ext3:ro \
#     /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \

    # --overlay /scratch/yy2694/data/saycam/saycam_shuffle_5fps.sqf:ro \
