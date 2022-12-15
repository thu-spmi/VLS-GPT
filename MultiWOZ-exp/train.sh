# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
python train_semi.py -mode train\
 -cfg lr=1e-4\
 gradient_accumulation_steps=16\
 batch_size=2\
 epoch_num=50\
 cuda_device=$1\
 model_act=True\
 seed=22
