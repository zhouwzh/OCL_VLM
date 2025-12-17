singularity exec --nv \
    --overlay /scratch/wz3008/data/saycam/saycam_transcript_frames/saycam_transcript_5fps.sqf:ro \
    --overlay /scratch/wz3008/overlay-50G-10M-slotattn-test.ext3:ro \
    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh;conda activate steve; cd /scratch/wz3008/SlotAttn/steve/"

python eval_fgari_video.py \
    --data_path "/scratch/wz3008/cvcl-related/datasets/movi_e_eval/*" \
    --trained_model "/home/wz3008/steve/logs/2025-10-07T23:54:56.474332/best_model_at_epoch_90.pt"