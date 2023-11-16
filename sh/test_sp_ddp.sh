# CUDA_VISIBLE_DEVICES=0 tools/dist_train_sp_ddp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 1 --resume
CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp_ddp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume

# CUDA_VISIBLE_DEVICES=0,2 tools/dist_train_sp_ddp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-2048x2048.py 2 --resume

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ddp.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume

# CUDA_VISIBLE_DEVICES=0 tools/dist_train_ddp.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 1 --resume

