# singularity exec --nv \
#     --overlay /scratch/yy2694/data/saycam/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
#     --overlay /scratch/wz3008/overlay-50G-10M-metaslot.ext3:ro \
#     /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
#     -c "source /ext3/env.sh;conda activate MetaSlot; cd /scratch/wz3008/SlotAttn/MetaSlot; "

singularity exec --nv \
    --overlay /scratch/wz3008/overlay-50G-10M-metaslot.ext3:ro \
    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh;conda activate MetaSlot; cd /scratch/wz3008/SlotAttn/MetaSlot; "


# python train.py \
#     --data_dir  /saycam_transcript_5fps/ \
#     --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
#     --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/7-dinosaur-saycam-ViT_B16/42_20251023_235721/checkpoint.pth.tar \
#     --save_dir ./save \
#     --use_dp \
#     --dev \
#     --visualize

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --visualize \
    --visualize_ttt \
    --ttt_lr 2e-05 \
    --ttt_step 3

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --visualize \
    --visualize_ttt \
    --ttt_lr 2e-05 \
    --ttt_step 5

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --visualize \
    --visualize_ttt \
    --ttt_lr 6e-05 \
    --ttt_step 3

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --visualize \
    --visualize_ttt \
    --ttt_lr 6e-05 \
    --ttt_step 5

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --visualize \
    --visualize_ttt \
    --ttt_lr 8e-05 \
    --ttt_step 3

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --visualize \
    --visualize_ttt \
    --ttt_lr 8e-05 \
    --ttt_step 5

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --visualize \
    --visualize_ttt \
    --ttt_lr 1e-04 \
    --ttt_step 3

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --visualize \
    --visualize_ttt \
    --ttt_lr 1e-04 \
    --ttt_step 5

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --visualize \
    --visualize_ttt \
    --ttt_lr 0.00012 \
    --ttt_step 3

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --visualize \
    --visualize_ttt \
    --ttt_lr 0.00012 \
    --ttt_step 5


# #???
# python train.py \
#     --data_dir  / \
#     --cfg_file /home/yy2694/scratch/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
#     --ckpt_file /home/yy2694/scratch/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/42_20251021_234351/checkpoint.pth.tar \
#     --save_dir ./save \
#     --use_dp \
#     --dev \
#     --visualize

# #????
# python train.py \
#     --data_dir  / \
#     --cfg_file /home/yy2694/scratch/SlotAttn/MetaSlot/save/9-dinosaur-saycam-ViT_B16/42_20251023_030147/9-dinosaur-saycam-ViT_B16.py \
#     --ckpt_file /home/yy2694/scratch/SlotAttn/MetaSlot/save/9-dinosaur-saycam-ViT_B16/42_20251023_030147/checkpoint.pth.tar \
#     --save_dir ./save \
#     --use_dp \
#     --dev \
#     --visualize


singularity exec --nv \
    --overlay /scratch/yy2694/data/saycam/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
    --overlay /scratch/wz3008/data/saycam_mask.sqf:ro \
    --overlay /scratch/wz3008/overlay-50G-10M-metaslot.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh;conda activate MetaSlot; cd /scratch/wz3008/SlotAttn/MetaSlot; "


python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/42_20251019_201055/checkpoint.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --ttt_step 6 \
    --eval_fgari

# 39.69

unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE

python train.py \
    --data_dir  /saycam_transcript_5fps/ \
    --cfg_file /scratch/wz3008/SlotAttn/MetaSlot/save/dinosaur-saycam-ViT_B16/max_slot-7/7-dinosaur-saycam-ViT_B16.py \
    --ckpt_file /scratch/wz3008/SlotAttn/MetaSlot/checkpoint/checkpoint_21.pth.tar \
    --save_dir ./save \
    --use_dp \
    --dev \
    --ttt_step 6 \
    --eval_fgari