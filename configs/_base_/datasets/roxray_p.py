# dataset settings
dataset_type = 'ROXrayDataset_P'
data_root = '/media/datasets/roxray/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'simulated', 'angles'])
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
test_for_angle_pipeline = [
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
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'simulated', 'angles'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'train/annotations_patched/',
        # img_prefix=data_root + 'train/images_patched/',
        ann_file=data_root + 'train/annotations_patched_3/',
        # ann_file=data_root + 'train/annotations_error/',
        img_prefix=data_root + 'train/images_patched_3/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/annotations_p/',
        img_prefix=data_root + 'test/images/',
        # ann_file=data_root + 'test/annotations_patched_new2/',
        # img_prefix=data_root + 'test/images_patched_2/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'test/annotations_p/',
        # img_prefix=data_root + 'test/images/',
        ann_file=data_root + 'test/annotations_patched_3/',
        img_prefix=data_root + 'test/images_patched_3/',
        # ann_file=data_root + 'test/annotations_patched_test/',
        # img_prefix=data_root + 'test/images_patched_test/',
        # ann_file=data_root + 'test/annotations_pred',
        # img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline),
    test_for_angle=dict(
        type=dataset_type,
        ann_file=data_root + 'test/annotations_patched_3/',
        img_prefix=data_root + 'test/images_patched_3/',
        # ann_file=data_root + 'train/annotations_patched_new3/',
        # img_prefix=data_root + 'train/images_patched_3/',
        pipeline=test_for_angle_pipeline)
    )
