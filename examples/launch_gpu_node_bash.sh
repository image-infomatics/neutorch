#!/bin/bash

module load slurm
module load cuda
module load cudnn

<<<<<<< HEAD
CORES=8
=======
CORES=16
>>>>>>> parent of 874d7d6 (changes prior to merging with main)

srun -p gpu --gpus 2  --cpus-per-gpu=$CORES --pty bash -i

