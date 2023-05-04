#!/bin/bash

module load slurm
module load cuda
module load cudnn

CORES=8

#neutrain-pre --config-file ./config.yaml
srun -p gpu --gpus 2 -C v100 --cpus-per-gpu=$CORES --pty bash -i

