_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# ---------------------------------------------------------------------------
# SegFormer-B0 / AI Hub 차선 3클래스 — BEV(조감도) 버전
#
# 데이터: UTIL/aihub_make_bev.py -> UTIL/aihub_make_split.py
#         train 18,094 / val 4,461 / test 6,423 장 (512x768)
#         split 배정은 전방 시점(CAR_LANE_SEG/split_holdout)과 100% 동일 (--mirror-split).
#         두 모델의 차이는 오직 BEV 변환 여부다.
#
# 주의: SegFormerHead.linear_fuse 가 SyncBN 을 하드코딩해 GPU 1장이어도 분산 런처 필수.
#           ./tools/dist_train.sh <이 파일> 4
# ---------------------------------------------------------------------------

norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

# 클래스 가중치 — BEV train 18,094장 전수 픽셀 실측에서 sqrt-inverse frequency (최솟값=1)
#   background 6,849,227,581 (96.2666%) / solid 143,196,280 (2.0126%) / dashed 122,426,443 (1.7207%)
#
# BEV 는 하늘·건물을 잘라내고 도로만 확대하므로 전경 비율이 크게 오른다.
#   전방 시점 전경 1.62% -> BEV 3.73%   (SkyScapes 는 0.45%)
# 그 결과 손실의 배경:전경 기여도가 훨씬 완만해진다.
#   가중치 없음 25.8:1  ->  sqrt-inverse 3.6:1   (전방 시점은 60.4:1 -> 5.5:1 이었다)
#
# 4000 iter 를 넘겨도 solid/dashed IoU 가 0 근처면 median-frequency 로 교체한다.
#   class_weight = [0.02863, 1.36951, 1.60185]
class_weight = [1.0, 6.916, 7.4797]

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
dataset_type = 'AIHubLaneDataset'          # 클래스·확장자가 같아 전방 시점과 공용
data_root = '/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE_BEV/split_holdout'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# BEV 는 이미 512x768 이라 크롭 없이 통째로 쓴다.
crop_size = (768, 512)   # (h, w)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # BEV 는 지면 좌표계라 이미 스케일이 정규화돼 있다. 전방 시점(0.5~2.0)보다 좁게 준다.
    dict(type='Resize', img_scale=(512, 768), ratio_range=(0.75, 1.5)),
    # cat_max_ratio 를 켜면(<1.0) "한 클래스가 X% 넘으면 재크롭"을 10회 재시도하는데,
    # 배경이 96.3% 라 절대 만족될 수 없어 매 샘플마다 10번 헛돈다. 반드시 1.0(비활성).
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),     # BEV 에서도 좌우 대칭은 유효하다
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
        img_scale=(512, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    # 입력 화소가 전방 시점(1024x640)의 60% 라 배치를 키울 수 있다. 실측 후 조정.
    samples_per_gpu=8,
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
# 전방 시점은 유효배치 16 x 40,000 iter = 640,000 샘플이었다. 같은 노출을 맞추려면
# 유효배치 32 에서는 20,000 iter 면 된다. 다만 전방 시점이 마지막 평가에서 최고점을
# 찍어 수렴 전이었으므로 40,000 을 유지해 여유를 둔다 (샘플 기준 2배 노출).
runner = dict(type='IterBasedRunner', max_iters=40000)

# mmseg 0.11.0 의 EvalHook 은 best 체크포인트를 저장하지 않는다. max_keep_ckpts 를 걸면
# 중간 최고 성능이 지워질 수 있어 전부 보관한다 (B0 는 1개 45MB).
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

# 유효배치가 16 -> 32 로 2배이므로 lr 도 2배 (1.2e-4 -> 2.4e-4)
optimizer = dict(_delete_=True, type='AdamW', lr=0.00024, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=3000,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
