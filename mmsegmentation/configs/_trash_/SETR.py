_base_ = [
    './_base_/models/setr_pup.py',
    './_base_/datasets/TrashDataset.py',
    './_base_/default_runtime.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (768, 768)
model = dict(
    pretrained=None,
    backbone=dict(
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrain/vit_large_p16.pth')),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=11,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=11,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=11,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            num_convs=2,
            up_scale=4,
            kernel_size=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512, 512)))

# optimizer
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={'head': dict(lr_mult=10.)}))

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=0.0,
    by_epoch=False)

checkpoint_config = dict(interval=5)
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4)

resume_from = '/opt/ml/input/mmsegmentation/work_dirs/epoch_25.pth'
# load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_vit-large_8x1_768x768_80k_cityscapes/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth'