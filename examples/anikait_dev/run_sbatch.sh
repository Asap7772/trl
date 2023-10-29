#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes, optionally iris-hi
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=16 # Request 8 CPUs for this task
#SBATCH --mem=48G # Request 32GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name=sft_trl # Name the job (for easier monitoring)
#SBATCH --account=iris
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hp-z8

python /iris/u/asap7772/trl/examples/anikait_dev/sft.py