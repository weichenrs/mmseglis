_base_ = [
    '../_base_/models/my_upernet_vit-b16_ln_mln.py',
    # '../_base_/datasets/fbp_1024x1024_crop_sp.py', 
    '../_base_/datasets/fbp_1024x1024_crop.py', 
    # '../_base_/default_runtime.py',
    '../_base_/wandb_runtime.py',
    # '../_base_/schedules/sp_schedule_80k.py'
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (1024, 1024)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    )

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/vit_base_patch16_224.pth',
    backbone=dict(img_size=crop_size, 
                  with_cp=True,
                  ),
    decode_head=dict(num_classes=24),
    auxiliary_head=dict(num_classes=24),
    # test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(896, 896))
    # test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(384, 384))
    test_cfg=dict(mode='whole')
    )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone

strategy = dict(
    type='DeepSpeedStrategy',
    fp16=dict(
        enabled=True,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    # inputs_to_half=[0],
    inputs_to_half=['inputs'],
    zero_optimization=dict(
        stage=3,
        allgather_partitions=True,
        reduce_scatter=True,
        allgather_bucket_size=50000000,
        reduce_bucket_size=50000000,
        overlap_comm=True,
        contiguous_gradients=True,
        cpu_offload=False),
)

# runner_type = 'FlexibleRunner'
optim_wrapper = dict(
    _delete_=True,
    # type='DeepSpeedOptimWrapper',
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    # accumulative_counts=2
    )

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

# vis_backends = [dict(type='WandbVisBackend')]
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=5)

# default_hooks = dict(
    # timer=dict(type='IterTimerHook'),
    # logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    # param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=15, 
    #                 max_keep_ckpts=5, save_best='mIoU'),
    # sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='SegVisualizationHook')
    # )
