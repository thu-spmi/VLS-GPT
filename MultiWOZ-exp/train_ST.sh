# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script runs the Semi-ST (self training) experiment on MultiWOZ.
# Before Semi-ST, make sure that you have pretrained your model on supervised data.
# The model in path will be loaded as initialization model.
# The proportion you choose must be consistent with that during pretraining stage.

ratio=$2
seed=22
path=experiments_21/all_pre_${ratio}_act_sd${seed}_lr0.0001_bs2_ga16/best_score_model
python train_semi.py \
    -mode semi_ST \
    -cfg  lr=1e-4 \
    seed=$seed\
    gradient_accumulation_steps=16 batch_size=2 \
    epoch_num=40 \
    cuda_device=$1 \
    gpt_path=$path \
    spv_proportion=$ratio\
    straight=True\
    ST_resp_only=True
