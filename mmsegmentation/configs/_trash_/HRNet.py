_base_ = [
    './_base_/models/fcn_hr18.py',
    './_base_/datasets/TrashDataset.py',
    './_base_/default_runtime.py', 
]

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384))
        )
    ),
    decode_head=dict(
        num_classes=11,
        in_channels=[48, 96, 192, 384], 
        channels=sum([48, 96, 192, 384])
    )
)

# optimizer
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)

checkpoint_config = dict(interval=5)
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')

resume_from = '/opt/ml/input/mmsegmentation/work_dirs/HRNet/best_mIoU63_epoch_67.pth'
# load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/hrnet/fcn_hr48_512x512_20k_voc12aug/fcn_hr48_512x512_20k_voc12aug_20200617_224419-89de05cd.pth'