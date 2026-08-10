#!/usr/bin/env python3
"""AI Hub 차선 데이터를 mmseg CustomDataset 레이아웃으로 분할한다.

## 반드시 클립 단위로 나눠야 하는 이유 (실측)

이 데이터는 주행 영상에서 뽑은 **연속 프레임**이다. 파일명이 연속 정수이고, 간격 1 인
쌍이 전체의 92% 다. 인접 프레임의 화소 평균 절대차는 20.8 인데 무작위 쌍은 62.2 로,
**인접 프레임이 무작위 쌍보다 3.0 배 유사**하다.

프레임 단위로 무작위 분할하면 0.03 초 차이 나는 사실상 같은 장면이 train 과 test 에
동시에 들어가 점수가 부풀려진다. 그래서 **연속 ID 구간(클립)을 통째로 한 split 에**
배정한다.

## 클립 분할만으로는 부족하다 — 서브셋 홀드아웃 (`--test-subsets`)

클립 단위 분할은 "완전히 같은 프레임"이 갈리는 것은 막지만, 그 이상은 못 막는다. 실측:

    서브셋 **내부** 클립 간 ID 거리 : 중앙값   118 ~    453   (300 이내 47~60%)
    서브셋 **간**      ID 거리      : 중앙값 7,768 ~ 43,260   (300 이내  1.7~11.4%)

같은 폴더 안의 클립들은 서로 몇 초 거리다. clip 71 과 clip 72 는 같은 도로를 몇 초
간격으로 지난 장면일 수 있다. 클립 단위로 나눠도 "같은 도로·같은 조명·같은 날씨"가
train 과 test 에 나뉘어 들어간다. 반면 다른 폴더끼리는 20~350 배 멀어 사실상 다른 주행이다.

그래서 **test 는 폴더(서브셋)째 홀드아웃**하는 편이 정직하다. `--test-subsets` 로 지정한
서브셋은 통째로 test 가 되고, 나머지 서브셋만 클립 단위로 train/val 로 나눈다.
val 이 클립 단위(다소 낙관적)인 것은 괜찮다 — val 은 체크포인트 선택용이고 정직한
숫자는 test 가 낸다.

## 층화(stratify) 방식

클립을 전경 픽셀량 순으로 정렬한 뒤 비율대로 라운드로빈 배정한다. 전경량이 큰 클립과
작은 클립이 고르게 퍼지므로, 무작위 배정보다 split 간 클래스 비율이 안정적이다.
결정론적이라 재실행해도 결과가 같다.

서브셋(주간/야간/해상도)은 각각 따로 층화한 뒤 합친다. 그래야 각 split 에 주간·야간이
같은 비율로 들어간다.

## 출력 (mmseg CustomDataset 규약)

    {out}/images/{train,val,test}/{subset}__{stem}.jpg
    {out}/labels/{train,val,test}/{subset}__{stem}.png

이미지와 라벨의 파일명이 확장자만 빼고 동일해야 한다 (`custom.py:146-151` 의
`img.replace(img_suffix, seg_map_suffix)` 가 단순 문자열 치환이기 때문).
서브셋이 달라도 stem 이 겹칠 수 있어 `{subset}__{stem}` 으로 접두어를 붙인다.

원본을 복사하지 않고 **심볼릭 링크**로 만든다 (이미지 15.4GB 중복 방지).
"""

import argparse
import collections
import csv
import os
import sys
from pathlib import Path

DEFAULT_SRC = Path("/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE_SEG")
SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_SRC / "manifest.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_SRC / "split")
    ap.add_argument("--ratio", nargs=3, type=int, default=[6, 2, 2],
                    help="train val test 비율 (--test-subsets 없을 때만 사용)")
    ap.add_argument("--test-subsets", nargs="+", default=None,
                    help="이 서브셋들을 통째로 test 로 홀드아웃한다. 나머지는 --train-val-ratio 로 나눔")
    ap.add_argument("--train-val-ratio", nargs=2, type=int, default=[8, 2],
                    help="--test-subsets 사용 시 나머지 서브셋의 train:val 비율")
    ap.add_argument("--mirror-split", type=Path, default=None,
                    help="기존 split 디렉토리의 클립 배정을 그대로 복제한다. "
                         "BEV 등 다른 전처리로 만든 데이터를 원본과 공정 비교할 때 쓴다 "
                         "(전경 픽셀량이 달라지면 층화 정렬 순서가 바뀌어 배정이 달라지므로)")
    ap.add_argument("--copy", action="store_true", help="심볼릭 링크 대신 실제 복사")
    ap.add_argument("--drop-empty-train", action="store_true",
                    help="전경 0 인 프레임을 train 에서 제외 (val/test 는 오탐 측정용으로 유지)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vis", type=int, default=30,
                    help="test 무작위 N장의 라벨 시각화를 {out}/vis 에 생성. 0 이면 생략. "
                         "WORKSPACE_ORGANIZATION_GUIDE 8.1절에 따라 분할마다 필수")
    ap.add_argument("--vis-seed", type=int, default=0, help="시각화 표본 seed (재실행 시 동일 장)")
    ap.add_argument("--vis-orig",
                    default="/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE/[원천]{subset}/{stem}.jpg",
                    help="변환 전 원본 이미지 경로 템플릿 ({subset}, {stem} 치환). "
                         "지정하면 `원본 | 변환후 | 라벨` 3분할이 된다. 빈 문자열이면 2분할")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    for r in rows:
        for k in ("clip", "solid_px", "dashed_px", "H", "W"):
            r[k] = int(r[k])
    print(f"manifest {len(rows):,}장")

    # (subset, clip) 을 하나의 단위로 묶는다
    clips = collections.defaultdict(list)
    for r in rows:
        clips[(r["subset"], r["clip"])].append(r)

    all_subs = sorted({r["subset"] for r in rows})
    held = set(args.test_subsets or [])
    unknown = held - set(all_subs)
    if unknown:
        print(f"[!] 존재하지 않는 서브셋: {sorted(unknown)}")
        print(f"    사용 가능: {all_subs}")
        return 1

    # 기존 split 의 배정을 그대로 따라간다 (공정 비교용)
    mirror = None
    if args.mirror_split:
        mirror = {}
        for sp in SPLITS:
            d = args.mirror_split / "labels" / sp
            if not d.is_dir():
                print(f"[!] {d} 없음")
                return 1
            for f in os.listdir(d):
                sub, stem = f.rsplit(".", 1)[0].rsplit("__", 1)
                mirror[(sub, stem)] = sp
        print(f"\n기존 split 복제: {args.mirror_split}  ({len(mirror):,}장의 배정을 그대로 사용)")

    if held:
        pat = ["train"] * args.train_val_ratio[0] + ["val"] * args.train_val_ratio[1]
        print(f"\n서브셋 홀드아웃: test = {sorted(held)}")
        print(f"나머지 {sorted(set(all_subs) - held)} 를 클립 단위 "
              f"{args.train_val_ratio[0]}:{args.train_val_ratio[1]} 로 train/val 분할")
    else:
        pat = []
        for i, s in enumerate(SPLITS):
            pat += [s] * args.ratio[i]

    # 서브셋별로 층화 배정
    assign = {}
    per_sub = collections.defaultdict(list)
    for key, rs in clips.items():
        per_sub[key[0]].append((key, sum(x["solid_px"] + x["dashed_px"] for x in rs)))
    if mirror is not None:
        # 클립 안의 프레임은 모두 같은 split 이므로 대표 1장으로 결정한다
        missing = 0
        for key, rs in clips.items():
            sp = next((mirror.get((r["subset"], r["stem"])) for r in rs
                       if (r["subset"], r["stem"]) in mirror), None)
            if sp is None:
                missing += 1
                sp = "train"
            assign[key] = sp
        if missing:
            print(f"[!] 기존 split 에 없어 train 으로 보낸 클립 {missing}개")
    else:
        for sub, lst in per_sub.items():
            if sub in held:                            # 통째로 test
                for key, _ in lst:
                    assign[key] = "test"
                continue
            lst.sort(key=lambda t: (-t[1], t[0]))      # 전경량 내림차순, 동점은 키순 -> 결정론적
            for i, (key, _) in enumerate(lst):
                assign[key] = pat[i % len(pat)]

    # 집계
    stat = {s: collections.Counter() for s in SPLITS}
    sub_stat = collections.defaultdict(lambda: collections.Counter())
    out_rows = collections.defaultdict(list)
    for key, rs in clips.items():
        sp = assign[key]
        for r in rs:
            if sp == "train" and args.drop_empty_train and r["solid_px"] + r["dashed_px"] == 0:
                continue
            out_rows[sp].append(r)
            stat[sp]["n"] += 1
            stat[sp]["solid"] += r["solid_px"]
            stat[sp]["dashed"] += r["dashed_px"]
            stat[sp]["px"] += r["H"] * r["W"]
            sub_stat[sp][r["subset"]] += 1
        stat[sp]["clips"] += 1

    print(f"\n{'split':6s} {'클립':>6s} {'장수':>8s} {'solid%':>8s} {'dashed%':>8s} {'전경%':>8s}")
    for s in SPLITS:
        c = stat[s]
        sp_, dp_ = c["solid"] / c["px"] * 100, c["dashed"] / c["px"] * 100
        print(f"{s:6s} {c['clips']:6,d} {c['n']:8,d} {sp_:7.3f}% {dp_:7.3f}% {sp_+dp_:7.3f}%")
    print("\n서브셋 분포")
    for s in SPLITS:
        tot = sum(sub_stat[s].values())
        print(f"  {s:6s} " + "  ".join(f"{k[:22]}={v:,}({v/tot*100:.0f}%)" for k, v in sorted(sub_stat[s].items())))

    # 클립 누수 검증
    seen = {}
    leak = 0
    for key, sp in assign.items():
        if key in seen and seen[key] != sp:
            leak += 1
        seen[key] = sp
    per_split_clips = collections.defaultdict(set)
    for key, sp in assign.items():
        per_split_clips[sp].add(key)
    inter = 0
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                inter += len(per_split_clips[a] & per_split_clips[b])
    print(f"\n클립 누수 검사: split 간 공유 클립 {inter}개 (0이어야 정상)")
    assert inter == 0

    if args.dry_run:
        print("\n[dry-run] 파일 생성 생략")
        return 0

    link = os.symlink if not args.copy else None
    import shutil
    for s in SPLITS:
        for kind in ("images", "labels"):
            d = args.out / kind / s
            if d.exists() and args.overwrite:
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in SPLITS:
        for r in out_rows[s]:
            name = f"{r['subset']}__{r['stem']}"
            for kind, src, ext in (("images", r["img"], ".jpg"), ("labels", r["mask"], ".png")):
                dst = args.out / kind / s / (name + ext)
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    os.symlink(src, dst)
            n += 1
    print(f"\n생성 완료 {n:,}쌍 -> {args.out}")
    print("주의: 이미지 확장자가 .jpg, 라벨이 .png 라 config 에서")
    print("      img_suffix='.jpg', seg_map_suffix='.png' 로 지정해야 한다.")

    if args.vis:
        make_vis(args.out, args.vis, args.vis_seed, args.vis_orig or None)
    return 0


# 클래스 색 (BGR). 데이터셋 PALETTE 계승: solid=파랑, dashed=빨강
VIS_COLORS = {1: (255, 0, 0), 2: (0, 0, 255)}
VIS_NAMES = {1: "solid", 2: "dashed"}


def make_vis(out_dir, n, seed, orig_tmpl=None, panel_h=760):
    """test 무작위 N장의 라벨 시각화를 만든다.

    `orig_tmpl` 이 있으면 `원본(변환 전) | 변환 후 | 라벨` 3분할, 없으면 `이미지 | 라벨` 2분할.
    BEV 처럼 전처리로 모습이 크게 바뀌는 데이터는 3분할이라야 변환이 맞았는지 판단할 수 있다.

    라벨 변환은 조용히 틀린다 (좌표계 뒤집힘, 클래스 ID 밀림, 래스터화 오류). 파일은 정상적으로
    만들어지고 학습도 돌아가므로 몇 시간 태운 뒤에야 알게 된다. 30장만 보면 바로 잡힌다.
    WORKSPACE_ORGANIZATION_GUIDE 8.1절 참조.
    """
    import random

    import cv2
    import numpy as np

    idir, adir = out_dir / "images" / "test", out_dir / "labels" / "test"
    names = sorted(os.listdir(idir))
    if not names:
        print("[!] test 가 비어 있어 vis 생략")
        return
    random.seed(seed)
    pick = random.sample(names, min(n, len(names)))
    vdir = out_dir / "vis"
    vdir.mkdir(parents=True, exist_ok=True)
    for f in vdir.iterdir():
        if f.is_file():
            f.unlink()

    def fit(im):
        """패널 높이를 맞춘다. 원본과 변환본의 종횡비가 달라도 나란히 놓이게."""
        w = max(1, int(round(im.shape[1] * panel_h / im.shape[0])))
        return cv2.resize(im, (w, panel_h), interpolation=cv2.INTER_AREA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    made = n_orig = 0
    for name in pick:
        stem_full = name[:-4]
        img = cv2.imread(str(idir / name))
        m = cv2.imread(str(adir / (stem_full + ".png")), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue

        # 얇은 선은 축소하면 사라진다. 축소 배율만큼 팽창시키고 그 값을 이미지에 적는다.
        th = max(0, int(round(img.shape[0] / panel_h)) - 1)
        ov = img.astype(np.float32)
        k = np.ones((2 * th + 1, 2 * th + 1), np.uint8) if th else None
        cnt = {}
        for c, col in VIS_COLORS.items():
            b = (m == c).astype(np.uint8)
            cnt[c] = int(b.sum())
            if not cnt[c]:
                continue
            if k is not None:
                b = cv2.dilate(b, k)
            bb = b.astype(bool)
            ov[bb] = 0.42 * ov[bb] + 0.58 * np.array(col, np.float32)
        ov = ov.astype(np.uint8)

        panels, titles = [], []
        if orig_tmpl and "__" in stem_full:
            sub, stem = stem_full.rsplit("__", 1)
            op = orig_tmpl.format(subset=sub, stem=stem)
            o = cv2.imread(op)
            if o is not None:
                panels.append(fit(o))
                titles.append(f"original (before)  {stem_full[:46]}")
                n_orig += 1
        panels.append(fit(img))
        titles.append("after transform" if panels and len(panels) > 1 else f"image  {stem_full[:46]}")
        tag = f"label  solid {cnt[1]:,} / dashed {cnt[2]:,}"
        if th:
            tag += f"   (보기용 {2 * th + 1}px 굵게 - 실제 두께 아님)"
        panels.append(fit(ov))
        titles.append(tag)

        pad, head, foot = 10, 40, 34
        W = sum(p.shape[1] for p in panels) + pad * (len(panels) + 1)
        canvas = np.full((panel_h + head + foot, W, 3), 22, np.uint8)
        x = pad
        for p, t in zip(panels, titles):
            canvas[head:head + panel_h, x:x + p.shape[1]] = p
            cv2.putText(canvas, t, (x + 2, head - 13), font, 0.58, (238, 238, 244), 2, cv2.LINE_AA)
            x += p.shape[1] + pad
        y = head + panel_h + 6
        for i, c in enumerate(VIS_COLORS):
            xx = 14 + i * 180
            cv2.rectangle(canvas, (xx, y + 3), (xx + 24, y + 19), VIS_COLORS[c], -1)
            cv2.putText(canvas, VIS_NAMES[c], (xx + 32, y + 18), font, 0.55,
                        (230, 230, 236), 1, cv2.LINE_AA)
        cv2.imwrite(str(vdir / f"{stem_full}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        made += 1
    extra = f", 원본 패널 포함 {n_orig}장" if orig_tmpl else ""
    print(f"vis: test 무작위 {made}장 (seed {seed}){extra} -> {vdir}")


if __name__ == "__main__":
    sys.exit(main())
