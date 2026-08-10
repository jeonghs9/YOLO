_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# ---------------------------------------------------------------------------
# SegFormer-B0 / DLR-SkyScapes 항공영상 차선 3클래스 (background, solid, dashed)
#
# 데이터: UTIL/skyscapes_make_split.py -> UTIL/skyscapes_make_tiles.py 로 만든
#         1024x1024 타일 (train 222, val 48, test 48)
# 백본  : UTIL/mit_hf_to_nvlabs.py 로 HuggingFace nvidia/mit-b0 를 변환한 것
#
# 주의: SegFormerHead.linear_fuse 가 SyncBN 을 하드코딩하고 있어서 GPU 가 1장이어도
#       반드시 분산 런처로 실행해야 한다. `python tools/train.py` 는 실패한다.
#           ./tools/dist_train.sh <이 파일> 1
# ---------------------------------------------------------------------------

norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

# 클래스 가중치 — train 타일 222장 실측 픽셀 수에서 계산 (sqrt-inverse frequency, 최솟값=1)
#   background 231,630,576 (99.5046%) / solid 787,720 (0.3384%) / dashed 365,576 (0.1570%)
# 가중치를 안 주면 손실의 배경:전경 기여도가 200.8 : 1 이라 전부 background 로 수렴한다.
# 아래 값이면 10.2 : 1 로 줄어든다.
#
# 학습이 4000 iter 를 넘겨도 solid/dashed IoU 가 0 근처라면 아래 median-frequency
# 가중치로 교체한다 (배경:전경 = 0.5 : 1 로 완전 균형. 대신 오탐이 늘 수 있다).
#   class_weight = [0.00323, 0.94993, 2.04684]
class_weight = [1.0, 17.1479, 25.1715]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b0.pth',
    backbone=dict(
        type='mit_b0',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=class_weight,
            loss_weight=1.0)),
    train_cfg=dict(),
    # 타일이 정확히 crop_size 와 같은 1024x1024 라 슬라이딩이 필요 없다.
    test_cfg=dict(mode='whole'))

# --- dataset ---------------------------------------------------------------
dataset_type = 'SkyScapesLaneDataset'
data_root = '/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split/tiles'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    # cat_max_ratio 를 켜면(<1.0) "한 클래스가 X% 넘으면 다시 crop" 을 10회 재시도하는데,
    # 배경이 99.5% 라 절대 만족될 수 없어 매 샘플마다 10번 헛돈다. 반드시 1.0(비활성).
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# V100 16GB 실측 peak VRAM (1024x1024, GPU 1장 기준)
#   samples_per_gpu=2 -> 6.80GB / =3 -> 10.19GB / =4 -> 13.58GB (총 15.77GB 중 86%)
# 4 는 평가 단계 오버헤드까지 겹치면 위험해서 2 로 둔다. 유효 배치는 GPU 장수로 키운다.
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        # 타일이 222장뿐이라 반복으로 감싸 dataloader 재시작 오버헤드를 없앤다.
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/images',
            ann_dir='train/labels',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/labels',
        pipeline=test_pipeline))

# --- schedule --------------------------------------------------------------
# 160k 는 Cityscapes(2975장) 기준이다. 여기선 타일 222장이라 20k 로 줄인다.
# GPU 4장 x batch 2 x 20000 iter = 160,000 샘플 = 타일 1장당 약 720회 노출.
runner = dict(type='IterBasedRunner', max_iters=20000)

# mmseg 0.11.0 의 EvalHook 은 best 체크포인트를 따로 저장하지 않는다. max_keep_ckpts 를
# 걸면 중간에 나온 최고 성능 체크포인트가 지워질 수 있어서 전부 보관한다 (B0 는 1개 15MB).
# 학습 후 log 의 mIoU 추이를 보고 쓸 체크포인트를 직접 고른다.
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
