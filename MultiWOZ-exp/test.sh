# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script tests the performance(BLEU, inform, success and combined score) of the model you choose
python train_semi.py -mode test -cfg gpt_path=$2  cuda_device=$1
