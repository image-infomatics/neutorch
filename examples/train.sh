#!/bin/bash

module load slurm
module load cuda
module load cudnn

export TF_CPP_MIN_LOG_LEVEL=1
export NCCL_DEBUG="INFO"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export TORCH_SHOW_CPP_STACKTRACES="1"

CORES_PER_GPU=8
NUM_TRAINERS=2
RANK=0
CONSTRAIN="a100"

#module list

#neutrain-pre --config-file ./config.yaml
#srun -p gpu --gpus 1 --cpus-per-gpu=$CORES neutrain-pre --config-file ./config.yaml
#python -m torch.distributed.launch --nproc_per_node=2 neutrain-affs-vol -c whole_brain_affs.yaml

#srun -p gpu --gpus $NUM_TRAINERS --cpus-per-gpu=$CORES_PER_GPU -C $CONSTRAIN torchrun \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node $NUM_TRAINERS \
    --no_python \
    neutrain-affs -c affs.yaml
    
#neutrain-affs-vol -c whole_brain_affs.yaml
    

#/mnt/home/jwu/code/neutorch/neutorch/train/whole_brain_affinity_map.py -c whole_brain_affs.yaml
