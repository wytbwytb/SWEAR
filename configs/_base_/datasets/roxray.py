# dataset settings
dataset_type = 'ROXrayDataset'
data_root = '/media/datasets/roxray/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

test_gt_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annotations/',
        img_prefix=data_root + 'train/images/',
        # ann_file=data_root + 'test/annotations/',
        # img_prefix=data_root + 'test/images/',
        # ann_file=data_root + 'train_hw/annotations/',
        # img_prefix=data_root + 'train_hw/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/annotations/',
        img_prefix=data_root + 'test/images/',
        # ann_file=data_root + 'test1_phase1/annotations/',
        # img_prefix=data_root + 'test1_phase1/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/annotations/',
        img_prefix=data_root + 'test/images/',
        # ann_file=data_root + 'test1_phase1/annotations/',
        # img_prefix=data_root + 'test1_phase1/images/',
        pipeline=test_pipeline),
    test_gt=dict(
        type=dataset_type,
        ann_file=data_root + 'test/annotations/',
        img_prefix=data_root + 'test/images/',
        # ann_file=data_root + 'test1_phase1/annotations/',
        # img_prefix=data_root + 'test1_phase1/images/',
        pipeline=test_gt_pipeline)
    )
