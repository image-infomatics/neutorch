#!/bin/bash

module load slurm
module load cuda
module load cudnn

CORES=12

#neutrain-pre --config-file ./config.yaml
srun -p gpu --gpus 1 --cpus-per-gpu=$CORES neutrain-pre --config-file ./config.yaml

