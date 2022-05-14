# Use the following command to request a single GPU node and one GPU
sintr -t 1:0:0 -N1 --gres=gpu:1 -A KRUEGER-SL2-GPU -p ampere