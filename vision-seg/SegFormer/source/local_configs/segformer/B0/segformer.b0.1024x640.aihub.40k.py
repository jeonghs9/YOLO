_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# ---------------------------------------------------------------------------
# SegFormer-B0 / AI Hub 차량 전방 차선 3클래스 (background, solid, dashed)
#
# 데이터: UTIL/aihub_make_masks.py -> UTIL/aihub_make_split.py
#         train 18,094 / val 4,461 / test 6,423 장 (1920x1200 또는 1920x1080)
# 백본  : UTIL/mit_hf_to_nvlabs.py 로 HuggingFace nvidia/mit-b0 를 변환한 것
#
# 주의: SegFormerHead.linear_fuse 가 SyncBN 을 하드코딩해 GPU 1장이어도 분산 런처 필수.
#           ./tools/dist_train.sh <이 파일> 4
# ---------------------------------------------------------------------------

norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

# 클래스 가중치 — train 18,094장 전수 픽셀 실측에서 sqrt-inverse frequency (최솟값=1)
#   background 41,009,124,016 (98.3702%) / solid 413,427,210 (0.9917%) / dashed 266,024,774 (0.6381%)
# 가중치 없으면 손실의 배경:전경 기여도가 60.4 : 1, 아래 값이면 5.5 : 1 로 줄어든다.
# (SkyScapes 는 각각 200.8 : 1 -> 10.2 : 1 이었다. 이 데이터가 전경이 3.6배 많아 훨씬 완만하다)
#
# 4000 iter 를 넘겨도 solid/dashed IoU 가 0 근처면 아래 median-frequency 로 교체한다.
#   class_weight = [0.01179, 1.16997, 1.81824]
class_weight = [1.0, 9.9596, 12.4159]

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
    test_cfg=dict(mode='whole'))

# --- dataset ---------------------------------------------------------------
dataset_type = 'AIHubLaneDataset'
data_root = '/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE_SEG/split_holdout'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 원본 1920x1200 을 가로 1024 로 맞추면 1024x640 (0.533배). 1920x1080 은 1024x576 이
# 되어 아래쪽이 Pad 로 채워진다(seg_pad_val=255 = ignore 라 손실·지표에서 빠진다).
# 하늘 영역 사전 크롭은 하지 않는다 — 추론 시 전처리를 단순하게 유지하기 위함.
crop_size = (640, 1024)   # (h, w)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 640), ratio_range=(0.5, 2.0)),
    # cat_max_ratio 를 켜면(<1.0) "한 클래스가 X% 넘으면 재크롭"을 10회 재시도하는데,
    # 배경이 98.4% 라 절대 만족될 수 없어 매 샘플마다 10번 헛돈다. 반드시 1.0(비활성).
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
        img_scale=(1024, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# 18,094장이라 SkyScapes 와 달리 RepeatDataset 이 필요 없다.
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='labels/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='labels/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='labels/test',
        pipeline=test_pipeline))

# --- schedule --------------------------------------------------------------
# GPU 4장 x batch 4 = 유효배치 16. 40,000 iter = 640,000 샘플 = 18,094장 기준 35 에폭.
# 단, 이 데이터는 연속 프레임이라 실제 장면은 1,419개뿐이다(중복 12.8배). 장면 기준으로는
# 451 회 노출에 해당한다.
runner = dict(type='IterBasedRunner', max_iters=40000)

# mmseg 0.11.0 의 EvalHook 은 best 체크포인트를 저장하지 않는다. max_keep_ckpts 를 걸면
# 중간 최고 성능이 지워질 수 있어 전부 보관한다 (B0 는 1개 45MB).
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

optimizer = dict(_delete_=True, type='AdamW', lr=0.00012, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=3000,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
