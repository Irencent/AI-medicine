_base_ = [
    '../../_base_/models/swin/swin_base.py', '../../_base_/default_runtime.py'
]

model=dict(type='Swinseg',
           backbone=dict(patch_size=(2,4,4),
                         embed_dim=128,
                         num_heads=[4, 8, 16, 32],
                         window_size=(8,7,7),
                         mlp_ratio=4.,
                         qkv_bias=True,
                         qk_scale=None,
                         drop_rate=0.,
                         attn_drop_rate=0.,
                         patch_norm=True,
                         drop_path_rate=0.3,
                         mode='seg'),
           num_frames=25,
           test_cfg=dict(max_testing_views=4),
           cls_head=dict(type='SegHead',num_classes=3,in_channels=32,dropout_ratio=0),
           )
# dataset settings
dataset_type = 'RawframeDataset'
data_root = ''
data_root_val = ''
ann_file_train = '/data/Joyce/VST/work_dir/annotations/seg/seg_ubk_sax_mid_2cls_area_train.txt'
ann_file_val = '/data/Joyce/VST/work_dir/annotations/seg/seg_ubk_sax_mid_2cls_area_val.txt'
ann_file_test = '/data/Joyce/VST/work_dir/annotations/seg/seg_ubk_sax_mid_2cls_area_testsub2k.txt'
# mask_ann = None
mask_ann = '/data/Joyce/UBK_VST/ubk_mask_ann.pkl'

img_norm_cfg_train = dict(
    mean=[87, 87, 87], std=[61, 61, 61], to_bgr=False)
img_norm_cfg_test = dict(
    mean=[87, 87, 87], std=[61, 61, 61], to_bgr=False)


padding = 120
train_pipeline = [
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=25, frame_interval=2, num_clips=1),
    dict(type='NIIDecodeV2', mask_ann=mask_ann,sax_concat=True),
    # dict(type='HistEqual'),
    # dict(type='DecordDecode'),
#    dict(type='Resize', scale=(-1, 256)),
#    dict(type='RandomResizedCrop'),
    dict(type='Padding', size=(padding, padding)),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='Random_rotate', range=20, ratio=0.5),
#     dict(type='Imgaug', transforms=[
#         dict(type='Rotate', rotate=(-20, 20))
#     ]),
    dict(type='ColorJitter', color_space_aug=True),
    #dict(type='Flip', flip_ratio=0.5),
    dict(type='AddRandomNumber', range=(-0.1, 0.1)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='SingleNorm'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'gt_semantic_seg'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'gt_semantic_seg'])
]
val_pipeline = [
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=25, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='NIIDecodeV2', mask_ann=mask_ann,sax_concat=True),
    #dict(type='HistEqual'),
    # dict(type='DecordDecode'),
#    dict(type='Resize', scale=(-1, 256)),
#    dict(type='RandomResizedCrop'),
    dict(type='Padding', size=(padding, padding)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='SingleNorm'),
#    dict(type='Normalize', **img_norm_cfg),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
    #dict(type='ColorJitter', color_space_aug=True),
    #dict(type='Imgaug', transforms=[
    #    dict(type='Rotate', rotate=(-90, 90))
    #]),
    #dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'gt_semantic_seg'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'gt_semantic_seg'])
]
test_pipeline = [
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=25, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='NIIDecodeV2', mask_ann=mask_ann,sax_concat=True),
    #dict(type='HistEqual'),
    # dict(type='DecordDecode'),
    dict(type='Padding', size=(padding, padding)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='SingleNorm'),
#    dict(type='Normalize', **img_norm_cfg),
    #dict(type='ThreeCrop', crop_size=224),
    #dict(type='ColorJitter', color_space_aug=True),
    #dict(type='Imgaug', transforms=[
    #    dict(type='Rotate', rotate=(-90, 90))
    #]),
    #dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'gt_semantic_seg'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'gt_semantic_seg'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=8,
    val_dataloader=dict(
        videos_per_gpu=12,
        workers_per_gpu=8
    ),
    test_dataloader=dict(
        videos_per_gpu=12,
        workers_per_gpu=8
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        task_type='Segmentation'),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        task_type='Segmentation'),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        task_type='Segmentation'))
evaluation = dict(
    interval=1, metrics=['mean_iou', 'mean_dice'])

# optimizer
optimizer = dict(type='AdamW', lr=3e-3, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))

# optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.001)


# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 100

# runtime settings
checkpoint_config = dict(interval=20)
work_dir = '/output/cmr'
find_unused_parameters = True

load_from = '/data/Joyce/VST/VST/checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
#resume_from = '/media/data/yanran/work_dirs/data_aug_ablation_study/None/epoch_50.pth'

# do not use mmdet version fp16
fp16 = None 
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=8,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)
