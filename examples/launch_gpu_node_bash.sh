#!/bin/bash

module load slurm
module load cuda
module load cudnn

CORES=16

srun -p gpu --gpus 2  --cpus-per-gpu=$CORES --pty bash -i

