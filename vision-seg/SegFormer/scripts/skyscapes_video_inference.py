#!/usr/bin/env python3
"""학습된 SegFormer 를 동영상에 적용해 `원본 | 예측` 비교 영상을 만든다.

주의 — 도메인 불일치:
  이 모델은 DLR-SkyScapes 의 **수직 하방(nadir) 항공영상**(GSD 약 13cm/px)으로 학습했다.
  비스듬한 CCTV/드론 영상에 적용하면 원근 왜곡과 스케일 차이 때문에 성능이 크게 떨어진다.
  결과를 정량 지표로 쓰면 안 되고, 전이 가능성을 눈으로 확인하는 용도로만 본다.

`--scale` 이 결과를 크게 좌우한다. test pipeline 의 `img_scale`(긴 변 기준)을 바꿔
입력 해상도를 조절하며, 이는 곧 "차선이 몇 픽셀 폭으로 보이는가"를 학습 시점(2~4px)에
맞추는 작업이다. 3840x2160 영상 실측:
  1024 (0.27x) -> 검출 가장 많음. 오탐도 많음        <- 기본값
  1920 (0.50x) -> 검출 적지만 점선 위치는 비교적 정확
  3840 (1.00x) -> 거의 검출되지 않음
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0]], np.uint8)
CLASS_NAMES = ("background", "solid", "dashed")
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SEG = Path("/home/hsjeong/workspace/vision-seg/SegFormer/source")


def get_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size, index=1)  # index 1 = Noto Sans CJK KR
    except Exception:
        return ImageFont.load_default()


def overlay(img_rgb, label, thicken=1, alpha=0.55):
    """예측 마스크를 반투명 오버레이로 얹는다.

    팽창은 반드시 cv2.dilate 를 쓴다. np.roll 반복으로 하면 4K 프레임에서
    (2*thicken+1)^2 회의 8M 배열 복사가 일어나 프레임당 0.7초를 잡아먹는다.
    """
    out = img_rgb.astype(np.float32)
    kernel = np.ones((2 * thicken + 1, 2 * thicken + 1), np.uint8)
    for c in (1, 2):
        m = (label == c).astype(np.uint8)
        if not m.any():
            continue
        if thicken > 0:
            m = cv2.dilate(m, kernel)
        mb = m.astype(bool)
        out[mb] = (1 - alpha) * out[mb] + alpha * PALETTE[c]
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default=str(SEG / "local_configs/segformer/B0/segformer.b0.1024x1024.skyscapes.20k.py"))
    ap.add_argument("--checkpoint", default=str(SEG / "work_dirs/b0_skyscapes_20k/iter_10000.pth"))
    ap.add_argument("--out", required=True, help="출력 mp4 경로")
    ap.add_argument("--scale", type=int, default=1024, help="추론 입력 긴 변 픽셀 (위 설명 참조)")
    ap.add_argument("--every", type=int, default=3, help="N 프레임마다 1장 처리 (3이면 30fps -> 10fps)")
    ap.add_argument("--panel-width", type=int, default=1280, help="패널 1개 가로 폭")
    ap.add_argument("--pred-only", action="store_true", help="원본 패널 없이 예측 오버레이만 출력")
    ap.add_argument("--thicken", type=int, default=2, help="축소 출력에서 선이 사라지지 않게 굵게")
    ap.add_argument("--max-frames", type=int, default=0, help="0 이면 전체")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from mmcv.utils import Config
    from mmseg.apis import inference_segmentor, init_segmentor

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    cfg = model.cfg.copy()
    cfg.data.test.pipeline[1]["img_scale"] = (args.scale, args.scale)
    model.cfg = cfg

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"영상을 열 수 없다: {args.video}")
        return 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    todo = (N + args.every - 1) // args.every
    if args.max_frames:
        todo = min(todo, args.max_frames)
    print(f"입력: {W}x{H} {FPS:.1f}fps {N}프레임 -> {args.every}프레임마다 처리, 총 {todo}장")

    pw = args.panel_width
    ph = int(round(H * pw / W))
    header = 40
    # H.264 등 대부분의 코덱이 짝수 해상도를 요구한다
    out_w = (pw + 20) if args.pred_only else (pw * 2 + 30)
    out_h = ph + header + 34  # 34 = 하단 범례 높이. 20 이면 글자가 잘린다
    out_size = (out_w + out_w % 2, out_h + out_h % 2)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS / args.every, out_size)
    font = get_font(22)
    ck = Path(args.checkpoint).stem

    idx = done = 0
    tot_solid = tot_dashed = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and done >= args.max_frames):
            break
        if idx % args.every:
            idx += 1
            continue
        idx += 1

        pred = inference_segmentor(model, frame)[0].astype(np.uint8)
        n1, n2 = int((pred == 1).sum()), int((pred == 2).sum())
        tot_solid += n1
        tot_dashed += n2

        # 4K 상태로 오버레이하면 100MB float 배열 연산이 프레임당 0.4초를 먹는다.
        # 먼저 출력 크기로 줄이고(라벨은 최근접) 작은 해상도에서 합성한다.
        rgb_s = cv2.cvtColor(cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        pred_s = cv2.resize(pred, (pw, ph), interpolation=cv2.INTER_NEAREST)
        ov = overlay(rgb_s, pred_s, thicken=args.thicken)

        canvas = Image.new("RGB", out_size, (20, 20, 24))
        d = ImageDraw.Draw(canvas)
        if args.pred_only:
            canvas.paste(Image.fromarray(ov), (10, header))
            d.text((12, 10), f"{ck} (scale {args.scale})  {idx / FPS:6.1f}s   "
                             f"solid {n1:,} / dashed {n2:,}", fill=(238, 238, 242), font=font)
        else:
            canvas.paste(Image.fromarray(rgb_s), (10, header))
            canvas.paste(Image.fromarray(ov), (pw + 20, header))
            d.text((12, 10), f"원본  {idx / FPS:6.1f}s", fill=(238, 238, 242), font=font)
            d.text((pw + 22, 10), f"예측 {ck} (scale {args.scale})  solid {n1:,} / dashed {n2:,}",
                   fill=(238, 238, 242), font=font)
        y = header + ph + 2
        for i, c in enumerate((1, 2)):
            x = 12 + i * 170
            d.rectangle([x, y + 4, x + 24, y + 16], fill=tuple(int(v) for v in PALETTE[c]))
            d.text((x + 32, y), CLASS_NAMES[c], fill=(230, 230, 236), font=get_font(18))

        vw.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
        done += 1
        if done % 50 == 0 or done == todo:
            el = time.time() - t0
            print(f"  {done}/{todo}  {el / done:.3f}s/frame  남은시간 {(todo - done) * el / done / 60:.1f}분")

    vw.release()
    cap.release()
    print(f"\n처리 {done}프레임, 총 {time.time() - t0:.0f}초")
    print(f"프레임당 평균 검출: solid {tot_solid / max(done,1):,.0f}px  dashed {tot_dashed / max(done,1):,.0f}px")
    print(f"출력: {args.out}  ({out_size[0]}x{out_size[1]}, {FPS / args.every:.1f}fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
