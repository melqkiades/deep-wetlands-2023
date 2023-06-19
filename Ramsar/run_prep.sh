#!/bin/env bash

#SBATCH -A NAISS2023-22-123    # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-07:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=T4:1   # choosing no. GPUs and their type
#SBATCH -J container           # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# Specify the path to the container
CONTAINER=/cephyr/users/ezioc/Alvis/deep-wetlands-2023/container_4.sif

# Print the PyTorch version then exit
singularity exec $CONTAINER stdbuf -oL nohup python3 -u /cephyr/users/ezioc/Alvis/deep-wetlands-2023/ramsar/google_data_preparation.py >> /cephyr/users/ezioc/Alvis/deep-wetlands-2023/ramsar/prep.log 2>&1
