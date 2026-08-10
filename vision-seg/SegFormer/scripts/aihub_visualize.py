#!/usr/bin/env python3
"""AI Hub 차선 데이터에 대해 `원본 | 정답 GT | 예측` 3분할 이미지를 장당 하나씩 저장한다.

SkyScapes 용 `skyscapes_visualize_test.py` 는 타일 구조에 맞춰져 있어 재사용이 어렵다.
이쪽은 mmseg CustomDataset 레이아웃(`images/{split}`, `labels/{split}`)을 그대로 읽는다.

GT 와 예측은 원본 위에 반투명 오버레이로 그린다. 차선이 얇아 마스크만 보면 도로와의
위치 관계를 알 수 없기 때문이다.
  solid = 파랑 [0,0,255]   dashed = 빨강 [255,0,0]   (데이터셋 PALETTE 계승)

**출력은 등배율이 아니다.** 화면에 맞춰 축소하면 폭 3~6px 차선이 사라지므로 축소 배율의
역수만큼 선을 굵게 그린다(`--thicken auto`). 실제 예측 두께를 보려면 `--thicken 0` 에
`--panel-width 1920` 을 준다.

반드시 segformer_cu111 환경의 python 으로 실행한다.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PALETTE = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0]], np.uint8)  # RGB
CLASS_NAMES = ("background", "solid", "dashed")
IGNORE = 255
ALPHA = 0.55
FONT = cv2.FONT_HERSHEY_SIMPLEX
SEG = Path("/home/hsjeong/workspace/vision-seg/SegFormer/source")


def overlay(img_bgr, label, thicken):
    """label(0/1/2/255) 을 BGR 이미지 위에 반투명으로 얹는다."""
    out = img_bgr.astype(np.float32)
    k = np.ones((2 * thicken + 1, 2 * thicken + 1), np.uint8) if thicken else None
    for c in (1, 2):
        b = (label == c).astype(np.uint8)
        if not b.any():
            continue
        if k is not None:
            b = cv2.dilate(b, k)
        bb = b.astype(bool)
        out[bb] = (1 - ALPHA) * out[bb] + ALPHA * PALETTE[c][::-1].astype(np.float32)  # RGB->BGR
    m = label == IGNORE
    if m.any():
        out[m] *= 0.35
    return out.astype(np.uint8)


def compose(panels, titles, ph, header=46, pad=8):
    """패널들을 **높이 기준**으로 맞춰 가로로 잇는다.

    전방 시점 원본(1920x1200, 종횡비 1.6)과 BEV(512x768, 종횡비 0.67)를 나란히 놓아야 하는데
    폭을 맞추면 한쪽이 심하게 뭉개진다. 그래서 높이를 맞추고 폭은 각자 비율대로 둔다.
    """
    small = [cv2.resize(p, (max(1, int(round(p.shape[1] * ph / p.shape[0]))), ph),
                        interpolation=cv2.INTER_AREA) for p in panels]
    W = sum(p.shape[1] for p in small) + pad * (len(small) + 1)
    H = ph + header + pad + 34
    canvas = np.full((H, W, 3), 22, np.uint8)
    x = pad
    for p, t in zip(small, titles):
        canvas[header:header + ph, x:x + p.shape[1]] = p
        # 패널 폭보다 긴 제목은 잘라 옆 패널 제목과 겹치지 않게 한다 (BEV 패널이 좁다)
        scale = 0.62
        while scale > 0.34 and cv2.getTextSize(t, FONT, scale, 2)[0][0] > p.shape[1] - 8:
            scale -= 0.04
        while t and cv2.getTextSize(t, FONT, scale, 2)[0][0] > p.shape[1] - 8:
            t = t[:-1]
        cv2.putText(canvas, t, (x + 4, header - 14), FONT, scale, (238, 238, 244), 2, cv2.LINE_AA)
        x += p.shape[1] + pad
    y = header + ph + 6
    for i, c in enumerate((1, 2)):
        xx = pad + 4 + i * 190
        cv2.rectangle(canvas, (xx, y + 4), (xx + 26, y + 20), tuple(int(v) for v in PALETTE[c][::-1]), -1)
        cv2.putText(canvas, CLASS_NAMES[c], (xx + 34, y + 19), FONT, 0.56, (230, 230, 236), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(SEG / "local_configs/segformer/B0/segformer.b0.1024x640.aihub.40k.py"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE_SEG/split_holdout")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=100, help="split 당 표본 수 (0 이면 전체)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel-height", type=int, default=560, help="패널 높이 (폭은 종횡비대로)")
    ap.add_argument("--orig",
                    default="/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE/[원천]{subset}/{stem}.jpg",
                    help="첫 패널로 쓸 **변환 전 쌩 원본** 경로 템플릿 ({subset}, {stem} 치환). "
                         "BEV 처럼 전처리한 데이터는 이걸 줘야 원본과 대조가 된다. "
                         "빈 문자열이면 입력 이미지(전처리본)를 첫 패널로 쓴다")
    ap.add_argument("--thicken", default="auto", help="선 팽창 반경. auto 면 축소 배율에서 자동")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from mmseg.apis import inference_segmentor, init_segmentor

    print(f"모델 로드: {Path(args.checkpoint).name}")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    root = Path(args.data_root)
    ck = Path(args.checkpoint).stem

    for split in args.splits:
        idir, adir = root / "images" / split, root / "labels" / split
        names = sorted(os.listdir(idir))
        random.seed(args.seed)
        pick = names if args.n == 0 else random.sample(names, min(args.n, len(names)))
        odir = args.out / split
        odir.mkdir(parents=True, exist_ok=True)

        inter = np.zeros(3, np.int64)
        union = np.zeros(3, np.int64)
        for i, name in enumerate(pick, 1):
            img = cv2.imread(str(idir / name))
            gt = cv2.imread(str(adir / (name[:-4] + ".png")), cv2.IMREAD_GRAYSCALE)
            if img is None or gt is None:
                print(f"  [!] 읽기 실패: {name}")
                continue
            pred = inference_segmentor(model, str(idir / name))[0].astype(np.uint8)

            valid = gt != IGNORE
            for c in range(3):
                p, g = (pred == c) & valid, (gt == c) & valid
                inter[c] += np.logical_and(p, g).sum()
                union[c] += np.logical_or(p, g).sum()

            th = args.thicken
            th = max(0, int(round(img.shape[0] / args.panel_height)) - 1) if th == "auto" else int(th)
            gs, gd = int((gt == 1).sum()), int((gt == 2).sum())
            ps, pd = int((pred == 1).sum()), int((pred == 2).sum())
            cond = "night" if "night" in name else "day"

            # 첫 패널: 가능하면 전처리 전 쌩 원본. BEV 처럼 모습이 바뀌는 전처리는 이게 있어야
            # "원본에 있던 차선이 제대로 넘어왔는지"를 판단할 수 있다.
            first, first_t = img, f"input  [{cond}]  {name[:-4][:44]}"
            if args.orig and "__" in name[:-4]:
                sub, stem = name[:-4].rsplit("__", 1)
                o = cv2.imread(args.orig.format(subset=sub, stem=stem))
                if o is not None:
                    first, first_t = o, f"original (raw)  [{cond}]  {name[:-4][:40]}"

            canvas = compose(
                [first, overlay(img, gt, th), overlay(img, pred, th)],
                [first_t,
                 f"GT   s {gs:,} / d {gd:,}",
                 f"pred  s {ps:,} / d {pd:,}"],
                args.panel_height)
            cv2.imwrite(str(odir / f"{name[:-4]}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if i % 25 == 0 or i == len(pick):
                print(f"  {split}: {i}/{len(pick)}")

        iou = np.where(union > 0, inter / np.maximum(union, 1) * 100, np.nan)
        print(f"[{split}] 표본 {len(pick)}장 IoU  "
              f"bg {iou[0]:.2f} / solid {iou[1]:.2f} / dashed {iou[2]:.2f}  "
              f"전경평균 {np.nanmean(iou[1:]):.2f}   -> {odir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
