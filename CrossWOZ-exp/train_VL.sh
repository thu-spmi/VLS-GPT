ratio=$2
path1=experiments/all_pre_${ratio}_act_sd11_lr0.0001_bs8_ga4/best_score_model
path2=experiments/all_pre_pos${ratio}_act_sd11_lr0.0001_bs8_ga4/best_loss_model
python train_semi_cross.py\
    -mode semi_VL\
    -cfg PrioriModel_path=$path1 PosteriorModel_path=$path2\
    lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    epoch_num=40\
    cuda_device=$1\
    spv_proportion=$ratio