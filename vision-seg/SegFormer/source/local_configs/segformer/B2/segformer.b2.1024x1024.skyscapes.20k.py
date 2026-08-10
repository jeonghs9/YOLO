_base_ = ['../B0/segformer.b0.1024x1024.skyscapes.20k.py']

# ---------------------------------------------------------------------------
# SegFormer-B2 / DLR-SkyScapes 3클래스
# B0 config 를 그대로 상속하고 백본·디코더 폭과 배치 크기만 바꾼다.
# 데이터 파이프라인, 클래스 가중치, 스케줄은 B0 와 동일하다.
# ---------------------------------------------------------------------------

model = dict(
    pretrained='pretrained/mit_b2.pth',
    backbone=dict(type='mit_b2'),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        decoder_params=dict(embed_dim=768)))

# V100 16GB 실측: 1024x1024 에서 samples_per_gpu=2 는 OOM (13.3GB 할당 후 1.5GB 추가 실패).
# 1 로 둔다. 유효 배치를 키우려면 GPU 장수를 늘린다 (samples_per_gpu 는 GPU 1장당 값).
data = dict(samples_per_gpu=1)
