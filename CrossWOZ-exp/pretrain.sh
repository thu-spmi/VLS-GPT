python train_semi_cross.py -mode pretrain\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    epoch_num=40\
    cuda_device=$1\
    spv_proportion=$2\
    model_act=True\
    posterior_train=False\
    save_type=max_score