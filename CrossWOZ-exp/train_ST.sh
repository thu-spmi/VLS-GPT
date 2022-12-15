ratio=$2
path=experiments_21/all_pre_${ratio}_act_sd11_lr0.0001_bs4_ga8/best_score_model
python train_semi_cross.py\
    -mode semi_ST\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=8 batch_size=4\
    epoch_num=40\
    cuda_device=$1\
    gpt_path=$path\
    model_act=True\
    spv_proportion=$ratio