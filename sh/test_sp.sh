# tools/dist_train.sh configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py 4 --work-dir work_dirs/pspnet_r50-d8_4xb4-80k_ade20k-512x512

# CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py 4 --work-dir work_dirs/sp_vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512

# tools/dist_train_sp.sh configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-1024x1024.py 2

# tools/dist_train_sp.sh configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-1024x1024.py 2

# CUDA_VISIBLE_DEVICES=2,3 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py 2
# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_8xb2-80k_fbp-512x512.py 2
# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_8xb2-80k_fbp-1024x1024_accum2.py 2
# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/mask2former/sp_mask2former_vit-b-160k_fbp-512x512.py 2
# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/mask2former/sp_mask2former_vit-b-160k_fbp-1024x1024.py 2

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/mask2former/sp_mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-512x512.py 2

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/mask2former/sp_mask2former_vit-b-160k_fbp-1024x1024_crop.py 2 --resume

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_8xb2-80k_fbp-512x512.py 2 --resume
# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_8xb2-80k_fbp-512x512.py 2 --resume

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb12-80k_fbp-512x512.py 2 --resume

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb12-80k_fbp-512x512_test.py 2 --resume

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume

CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py \
#                         work_dirs/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_30000.pth 2 --show-dir show_dirs/test1017

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py \
#                         work_dirs/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_50000.pth 2 --show-dir show_dirs/test1018_ori

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh configs/vit/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512.py \
#                         work_dirs/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512/best_mIoU_iter_65000.pth 2 \
#                         --show-dir show_dirs/test1019/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512_best_mIoU_iter_65000

# CUDA_VISIBLE_DEVICES=2,3 tools/dist_test.sh configs/vit/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512.py \
#                         work_dirs/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512/best_mIoU_iter_65000.pth 2 \
#                         --show-dir show_dirs/test1019/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512_best_mIoU_iter_65000_whole

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-512x512.py \
#                         work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-512x512/best_mIoU_iter_80000_2card.pth 2 \
#                         --show-dir show_dirs/test1019/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-512x512_best_mIoU_iter_80000_2card
