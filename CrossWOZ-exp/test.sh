# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script tests the performance(BLEU, inform, success and combined score) of the model you choose
device=$1
path=experiments/all_pre_20_act_sd11_lr0.0001_bs8_ga4/best_score_model
python train_semi_cross.py -mode test -cfg gpt_path=$path cuda_device=$device  eval_batch_size=32