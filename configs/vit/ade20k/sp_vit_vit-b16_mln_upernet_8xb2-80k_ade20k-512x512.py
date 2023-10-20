_base_ = [
    '../_base_/models/sp_upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/ade20k_sp.py', '../_base_/default_runtime.py',
    # '../_base_/schedules/sp_schedule_80k.py'
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)

data_preprocessor = dict(
    type='SPSegDataPreProcessor',
    size=crop_size,
    )

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/vit_base_patch16_224_sp.pth',
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

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
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

# vis_backends = [dict(type='WandbVisBackend')]

# default_hooks = dict(
#     logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
# )

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))