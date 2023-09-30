# tools/dist_train.sh configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py 4 --work-dir work_dirs/pspnet_r50-d8_4xb4-80k_ade20k-512x512

# tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512.py 4 --work-dir work_dirs/vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512

# tools/dist_train.sh configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py 4 --work-dir work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640 --resume

tools/dist_train.sh configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-1024x1024.py 4 --resume
tools/dist_train.sh configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_fbp-512x512.py 4 --resume

