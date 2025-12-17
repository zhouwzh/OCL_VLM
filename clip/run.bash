singularity exec --nv \
    --overlay /scratch/yy2694/data/saycam/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
    --overlay /scratch/wz3008/overlay-50G-10M-slotattn-test.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh;conda activate MetaSlot; cd /scratch/wz3008/SlotAttn/; "

python train_clip.py \
    --gpus 4 \
    --data_config /scratch/wz3008/SlotAttn/clip/config/data_config.yaml \
    --model_config /scratch/wz3008/SlotAttn/clip/config/model_config.yaml \
    --train_config /scratch/wz3008/SlotAttn/clip/config/train_config.yaml \
    --arch metaslot \
    --exp_name topk \
    --augment_frames \
    --ckpt_file /scratch/wz3008/SlotAttn/clip/saved_checkpoints/2025-10-27-23-53-16_freeze_ori_config/mean_checkpoint.pth