# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
posterior=False
python train_semi_cross.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    seed=3\
    epoch_num=50\
    cuda_device=$1\
    model_act=True\
    posterior_train=$posterior\
    exp_no=baseline1
