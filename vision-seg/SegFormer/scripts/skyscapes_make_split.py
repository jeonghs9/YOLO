#!/usr/bin/env python3
"""SkyScapes-Lane(SS_Multi_Lane)을 3클래스(background/solid/dashed)로 병합하고 재분할한다.

원본은 13클래스(0=background, 1~12=노면표시)다. 실선/점선 판정만 필요하므로 다음과 같이 병합한다.

    0 background <- 0(Background), 4(Turn signs), 5(Other signs), 6(Plus sign),
                    7(Crosswalk), 8(Stop line), 9(Zebra zone), 10(No parking zone),
                    11(Parking space), 12(Other lane-markings)
    1 solid      <- 2(Long Line)
    2 dashed     <- 1(Dash Line), 3(Small dash line)

background는 semantic segmentation 구조상 제거할 수 없다(모든 픽셀이 클래스 하나를 가져야 함).
성능 보고 시 mIoU는 solid/dashed 2개만 평균낸다.

split은 병합 후 solid/dashed 픽셀량이 train:val:test = 6:2:2 비율에 가장 가깝게 배분되도록
전수 탐색으로 결정한다. 탐색은 결정론적이라 몇 번 실행해도 같은 결과가 나온다.

사용법:
    python3 /home/hsjeong/workspace/vision-seg/SegFormer/scripts/skyscapes_make_split.py --overwrite
    python3 .../skyscapes_make_split.py --dry-run     # 파일 생성 없이 split 계획과 통계만 출력
"""

import argparse
import itertools
import os
import shutil
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SRC = "/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/원본/SS_Multi_Lane/SS_Multi_Lane"
DST = "/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split"

# 라벨이 있는 split (원본 test 6장은 GT가 없다)
LABELED_SPLITS = ("train", "val")
UNLABELED_SRC_SPLIT = "test"

# 원본 13클래스 -> 3클래스 LUT. 인덱스가 원본 ID, 값이 새 ID.
CLASS_LUT = np.zeros(256, dtype=np.uint8)
CLASS_LUT[2] = 1            # Long Line        -> solid
CLASS_LUT[1] = 2            # Dash Line        -> dashed
CLASS_LUT[3] = 2            # Small dash line  -> dashed
# 나머지(0, 4~12)는 0(background) 유지

NEW_CLASSES = ("background", "solid", "dashed")
# 원본 색상 계승: Long Line 파랑, Dash Line 빨강
NEW_PALETTE = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0]], dtype=np.uint8)

N_TRAIN, N_VAL, N_TEST = 6, 2, 2
TARGET_SHARE = {"train": N_TRAIN / 10, "val": N_VAL / 10, "test": N_TEST / 10}


def collect_labeled():
    """라벨이 있는 이미지들의 (원본split, stem) 목록을 반환한다."""
    out = []
    for s in LABELED_SPLITS:
        d = os.path.join(SRC, s, "labels", "grayscale")
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".png"):
                out.append((s, fn[:-4]))
    return out


def load_stats(items):
    """이미지별 병합 후 클래스 픽셀 수를 센다. {(split, stem): np.array([bg, solid, dashed])}"""
    stats = {}
    for s, stem in items:
        p = os.path.join(SRC, s, "labels", "grayscale", stem + ".png")
        a = CLASS_LUT[np.array(Image.open(p))]
        stats[(s, stem)] = np.bincount(a.ravel(), minlength=3).astype(np.int64)
    return stats


def search_split(items, stats):
    """solid/dashed 픽셀 배분이 6:2:2 에 가장 가까운 조합을 전수 탐색한다.

    목적함수: 각 (split, 전경클래스) 조합의 픽셀 점유율과 목표 점유율의 최대 편차. 작을수록 좋다.
    동점 시 train 전경 픽셀이 많은 쪽, 그다음 사전순으로 결정한다.
    """
    total = np.sum([stats[k] for k in items], axis=0)
    best = None
    for val in itertools.combinations(sorted(items), N_VAL):
        rest = [x for x in sorted(items) if x not in val]
        for test in itertools.combinations(rest, N_TEST):
            train = tuple(x for x in rest if x not in test)
            groups = {"train": train, "val": val, "test": test}

            # 모든 split이 solid, dashed 를 하나 이상 가져야 한다
            sums = {g: np.sum([stats[k] for k in m], axis=0) for g, m in groups.items()}
            if any(sums[g][c] == 0 for g in groups for c in (1, 2)):
                continue

            dev = max(
                abs(sums[g][c] / total[c] - TARGET_SHARE[g])
                for g in groups for c in (1, 2)
            )
            key = (round(dev, 9), -int(sums["train"][1] + sums["train"][2]), train, val, test)
            if best is None or key < best[0]:
                best = (key, groups, sums, dev)
    if best is None:
        sys.exit("[중단] 조건을 만족하는 split 조합이 없다.")
    return best[1], best[2], best[3]


def write_split(groups, args):
    n = 0
    for out_split, members in groups.items():
        for sub in ("images", "labels/grayscale", "labels/rgb"):
            os.makedirs(os.path.join(args.dst, out_split, sub), exist_ok=True)
        for src_split, stem in members:
            src_img = os.path.join(SRC, src_split, "images", stem + ".jpg")
            src_lbl = os.path.join(SRC, src_split, "labels", "grayscale", stem + ".png")
            if not (os.path.isfile(src_img) and os.path.isfile(src_lbl)):
                sys.exit(f"[중단] 원본 파일 없음: {stem}")

            shutil.copy2(src_img, os.path.join(args.dst, out_split, "images", stem + ".jpg"))

            a = CLASS_LUT[np.array(Image.open(src_lbl))]
            Image.fromarray(a, mode="L").save(
                os.path.join(args.dst, out_split, "labels", "grayscale", stem + ".png"))
            Image.fromarray(NEW_PALETTE[a], mode="RGB").save(
                os.path.join(args.dst, out_split, "labels", "rgb", stem + ".png"))
            n += 1
    return n


def write_unlabeled(args):
    d = os.path.join(args.dst, "unlabeled", "images")
    os.makedirs(d, exist_ok=True)
    src_dir = os.path.join(SRC, UNLABELED_SRC_SPLIT, "images")
    n = 0
    for fn in sorted(os.listdir(src_dir)):
        shutil.copy2(os.path.join(src_dir, fn), os.path.join(d, fn))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    items = collect_labeled()
    print(f"라벨 보유 이미지: {len(items)}장\n")

    stats = load_stats(items)
    print("이미지별 병합 후 픽셀 수")
    print(f"  {'orig':6s} {'stem':28s} {'solid':>9s} {'dashed':>9s}")
    for k in items:
        print(f"  {k[0]:6s} {k[1][-8:]:28s} {stats[k][1]:9d} {stats[k][2]:9d}")

    groups, sums, dev = search_split(items, stats)
    total = np.sum([stats[k] for k in items], axis=0)

    print(f"\n선택된 split (최대 점유율 편차 {dev*100:.2f}%p, 목표 60/20/20)")
    print(f"  {'split':6s} {'매수':>4s} {'solid':>9s} {'점유':>7s} {'dashed':>9s} {'점유':>7s}  구성")
    for g in ("train", "val", "test"):
        s = sums[g]
        stems = ", ".join(f"{sp}/{st[-8:]}" for sp, st in groups[g])
        print(f"  {g:6s} {len(groups[g]):4d} {s[1]:9d} {100*s[1]/total[1]:6.2f}% "
              f"{s[2]:9d} {100*s[2]/total[2]:6.2f}%  {stems}")

    if args.dry_run:
        print("\n[dry-run] 파일을 생성하지 않았다.")
        return

    if os.path.isdir(args.dst) and os.listdir(args.dst):
        if not args.overwrite:
            sys.exit(f"\n[중단] 출력 디렉토리가 비어 있지 않다: {args.dst}\n"
                     f"       덮어쓰려면 --overwrite 를 붙여라.")
        print(f"\n[삭제] {args.dst}")
        shutil.rmtree(args.dst)

    n_lab = write_split(groups, args)
    n_unlab = write_unlabeled(args)
    print(f"\n라벨 {n_lab}장 + 무라벨 {n_unlab}장 = {n_lab + n_unlab}장 생성")
    print(f"출력: {args.dst}")
    print(f"클래스: {', '.join(f'{i}={c}' for i, c in enumerate(NEW_CLASSES))}")


if __name__ == "__main__":
    main()
