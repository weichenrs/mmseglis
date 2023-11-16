# CUDA_VISIBLE_DEVICES=2,3 tools/dist_train2.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa.py 2 --resume --amp \
#                          --work-dir work_dirs/exp_1113/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_1113

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train2.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_nods.py 2 --resume \
#                          --work-dir work_dirs/exp_1114/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_nods

# CUDA_VISIBLE_DEVICES=2,3 tools/dist_train2.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume \
#                         --work-dir work_dirs/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_withcp_ds




################################################ DONE ################################################

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa.py 2 --resume --amp \
#                          --work-dir work_dirs/exp_1113/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_1113

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test.py 2 --resume \
#                         --work-dir work_dirs/exp_1114/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test


################################################ OKOK ################################################

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-2048x2048_ds_test.py 2 --resume \
#                         --work-dir work_dirs/exp_1115/vit_vit-b16_mln_upernet_2xb2-80k_fbp-2048x2048_ds_test

################################################ DOIN ################################################

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa.py 2 --resume --amp \
#                         --work-dir work_dirs/exp_1115/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa_nopsp

CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa.py 2 --resume --amp \
                        --work-dir work_dirs/exp_1116/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa_psp

################################################ TODO ################################################

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa.py 2 --resume \
#                         --work-dir work_dirs/exp_1115/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa.py 2 --resume --amp \
#                          --work-dir work_dirs/exp_1113/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_1113