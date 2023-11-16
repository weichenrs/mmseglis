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

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume
# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb12-80k_fbp-512x512_test.py 2 --resume

# CUDA_VISIBLE_DEVICES=2 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb12-80k_fbp-512x512_test.py 1 --resume



# CUDA_VISIBLE_DEVICES=2,3 tools/dist_train_sp2.sh configs/vit/sp_vit_vit-l16_mln_upernet_2xb2-40k_fbp-1024x1024.py 2 --resume \
#                         --work-dir work_dirs/sp_vit_vit-l16_mln_upernet_2xb2-40k_fbp-1024x1024_tt


# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-l16_mln_upernet_2xb2-40k_fbp-1024x1024.py 2 --amp --resume \
#                         --work-dir work_dirs/sp_vit_vit-l16_mln_upernet_2xb2-40k_fbp-1024x1024_bs2_withcp_amp


# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb12-80k_fbp-512x512_test.py 2 --resume
# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_test.py 2 --resume
# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/my_vit_vit-l16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume \
#                         --work-dir work_dirs/my_vit_vit-l16_mln_upernet_2xb2-80k_fbp-1024x1024_bs4

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_colo.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_col.py 2 --resume \
#                         --work-dir work_dirs/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_withcp_col 

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_colo.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-2048x2048_col.py 2 --resume \
#                         --work-dir work_dirs/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-2048x2048_withcp_col 

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume --amp \
#                         --work-dir work_dirs/1108_test_cp_withcp_amp

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_nocp.py 2 --resume --amp\
#                         --work-dir work_dirs/1108_test_cp_nocp_amp

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_colo.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_col.py 2 --resume \
#                         --work-dir work_dirs/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_col_withcp

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_zero.py 2 --resume

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_accum2.py 2 --resume \
#                         --work-dir work_dirs/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_nocp_accum2

##TODO
# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-160k_fbp-1024x1024_accum2.py 2 --resume \
#                         --work-dir work_dirs/my_vit_vit-b16_mln_upernet_2xb2-160k_fbp-1024x1024_nocp_accum2

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds.py 2 --resume --amp

# CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test.py 4 --resume --amp \
#                          --work-dir work_dirs/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_card4

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test.py 2 --resume --amp \
#                          --work-dir work_dirs/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_card2

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_80k.py 2 --resume --amp \
#                          --work-dir work_dirs/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_card2_80k

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa.py 2 --resume --amp\
                        #  --work-dir work_dirs/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_card2_80k

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_nods.py 2 --resume \
#                          --work-dir work_dirs/exp1113/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_torch_1113

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa.py 2 --resume --amp \
#                          --work-dir work_dirs/exp_1113/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_1113

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test.py 2 --resume \
#                         --work-dir work_dirs/exp_1114/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test.py 2 --resume \
#                         --work-dir work_dirs/exp_1114/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-2048x2048_ds_test.py 2 --resume \
#                         --work-dir work_dirs/exp_1115/vit_vit-b16_mln_upernet_2xb2-80k_fbp-2048x2048_ds_test


# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test.py 2 --resume --amp


# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py 2 --resume \


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

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-1024x1024.py \
#                         work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-1024x1024/best_mIoU_iter_25000.pth 2 \
#                         --show-dir show_dirs/test1020/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-1024x1024/best_mIoU_iter_25000
                        
# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test_2.sh configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-1024x1024.py \
#                         work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-1024x1024/iter_140000.pth 2 \
#                         --show-dir show_dirs/test1020/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-1024x1024/iter_140000

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-512x512.py \
#                         work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-512x512/best_mIoU_iter_80000_2card.pth 2 \
#                         --show-dir show_dirs/test1-2-/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-512x512/best_mIoU_iter_80000_2card



# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py \
#                         work_dirs/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_35000.pth 2 \
#                         --show-dir show_dirs/test1021/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_35000

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh configs/vit/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py \
#                         work_dirs/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_35000.pth 2 \
#                         --show-dir show_dirs/test1021/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_35000

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test_sp.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py \
#                         work_dirs/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_65000.pth 2 \
#                         --show-dir show_dirs/test1023/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_65000

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test_sp2.sh configs/vit/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py \
#                         work_dirs/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_later/iter_80000.pth 2 \
#                         --show-dir show_dirs/test1102/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/iter_80000_val

                        # work_dirs/sp_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_later/best_mIoU_iter_65000.pth 2 \