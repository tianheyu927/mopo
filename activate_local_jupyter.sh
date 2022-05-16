#! /bin/bash

############################
# Activate Local Environment
# i.e. on login node
############################
module load miniconda/3
eval "$(conda shell.bash hook)"
conda activate ./.env_jupyter
