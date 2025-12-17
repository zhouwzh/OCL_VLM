module purge
singularity exec --fakeroot --nv \
    --bind /scratch/yy2694/data/saycam/5fps:/data/saycam/5fps:ro \
    --overlay /home/yy2694/overlay-50G-10M-slotcontrast.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash \
    -c "source /ext3/env.sh;"