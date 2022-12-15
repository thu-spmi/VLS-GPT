# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script pretrains model on the labeled data of MultiWOZ.
# You can choose any supervised proportion (ratio).
# If posterior is True, you'll pretrain the inference model
# else you'll pretrain the generative model.
python train_semi.py -mode pretrain \
    -cfg  lr=1e-4 \
    gradient_accumulation_steps=16 batch_size=2 \
    epoch_num=50 \
    seed=22\
    cuda_device=$1 \
    spv_proportion=$2 \
    model_act=True \
    posterior_train=True \
    save_type='min_loss'