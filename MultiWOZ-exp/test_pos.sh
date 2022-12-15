path=experiments_21/all_VL_20_12-12_sd33_lr0.0001_bs2_ga16/best_post_model
python train_semi.py -mode test_pos -cfg gpt_path=$path  cuda_device=$1
