python train_semi.py -mode pretrain \
    -cfg  lr=1e-4 \
    gradient_accumulation_steps=16 batch_size=2 \
    seed=$3 \
    epoch_num=50 \
    cuda_device=$1 \
    spv_proportion=$2 \
    posterior_train=False

ratio=$2
seed=$3
path1=experiments_21/all_pre_${ratio}_act_sd${seed}_lr0.0001_bs2_ga16/best_score_model
path2=experiments_21/all_pre_pos${ratio}_act_sd${seed}_lr0.0001_bs2_ga16/best_loss_model

python train_semi.py \
    -mode semi_ST \
    -cfg PrioriModel_path=$path1 PosteriorModel_path=$path2 \
    lr=1e-4 \
    seed=$seed \
    gradient_accumulation_steps=16 batch_size=2 \
    epoch_num=40 \
    cuda_device=$1 \
    spv_proportion=$ratio\
    ST_with_infer=True\
    ST_resp_only=False\
    straight=True\
    freeze_infer=False\
    exp_no=ST_infer_0728