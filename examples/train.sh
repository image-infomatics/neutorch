#!/bin/bash

module load slurm
module load cuda
module load cudnn

CORES=12
NUM_TRAINERS=1
RANK=0

#neutrain-pre --config-file ./config.yaml
#srun -p gpu --gpus 1 --cpus-per-gpu=$CORES neutrain-pre --config-file ./config.yaml
#python -m torch.distributed.launch --nproc_per_node=2 neutrain-affs-vol -c whole_brain_affs.yaml
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node $NUM_TRAINERS \
    --no_python \
    neutrain-affs-vol -c whole_brain_affs.yaml
    

#/mnt/home/jwu/code/neutorch/neutorch/train/whole_brain_affinity_map.py -c whole_brain_affs.yaml
