#!/usr/bin/env python3
"""AI Hub 전방 시점 데이터를 BEV(조감도)로 변환한다.

최종 적용 대상이 드론뷰라서, 지면 평면 정사영인 BEV 가 목표 도메인 그 자체다.
자세한 근거와 검증 수치는 `UTIL/md/SEGFORMER_AIHUB_02_BEV_PLAN.md` 참조.

## 호모그래피를 어떻게 얻는가

카메라 캘리브레이션이 전혀 없다(원본에 jpg/json 뿐). 그래서 **소실점을 데이터에서 추정**해
호모그래피를 만든다. 직선 도로에서 평행한 차선들은 이미지 상에서 소실점 한 점으로 수렴하므로,
차선 폴리라인들의 쌍별 교점 중앙값이 소실점이다.

실측 결과 서브셋별로 소실점이 일관적이라(1200 서브셋 3개는 MAD ±16~22px) **서브셋별 고정
호모그래피 1개**로 충분하다. 검증: BEV 에서 차선의 수직 기준 각도가 68도 -> 1.2~3.3도.

## 라벨은 마스크가 아니라 폴리라인을 변환한다

기존 마스크(원근 보정 폭)를 warp 하면 원거리에서 극단적으로 늘어나 계단·구멍이 생기고,
원근 보정한 폭을 다시 왜곡하는 이중 근사가 된다.

**폴리라인을 warp 한 뒤 BEV 에서 그린다.** BEV 에서는 차선 폭이 상수이므로 단일 폭으로
그리면 되고, 01단계의 원근 보정표가 아예 불필요해진다. BEV 로 가는 목적 자체가 원근 제거다.

## 소실선 위쪽 점 처리 (중요)

소실점보다 위(y < vy)에 있는 점은 호모그래피로 보내면 무한대로 발산하거나 부호가 뒤집혀
엉뚱한 곳에 찍힌다. 그래서 폴리라인을 촘촘히 리샘플한 뒤 **사다리꼴 위변(y_far) 아래 점만**
남기고 변환한다. 이렇게 하면 클리핑이 기하학적으로 올바르게 된다.
"""

import argparse
import csv
import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

SRC = Path("/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE")
DST = Path("/home/hsjeong/workspace/dataset/AIHUB/CAR_LANE_BEV")

CLASS_NAMES = ("background", "solid", "dashed")
LANE_TYPE_TO_ID = {"solid": 1, "dotted": 2}

# vp_estimate.py 실측 소실점 (서브셋별 중앙값)
VP = {
    "c_1920_1200_daylight_validation_2": (997.8, 571.9),
    "c_1920_1200_daylight_validation_3": (996.9, 573.0),
    "c_1920_1200_night_validation_1": (973.0, 593.0),
    "1920_1080_night_validation_d_1": (952.6, 568.9),
}

BEV_W, BEV_H = 512, 768
MARGIN = 0.30          # BEV 가로 폭 중 좌우 여백 비율 (도로가 화면 꽉 차지 않게)

# 사다리꼴 파라미터 — 실측 트레이드오프에서 고른 균형점 (daylight_3, 120장 표본)
#   half_w  y_far  검은영역%  전경%  프레임당차선
#     0.50     60      0.7    1.75      2.98
#     0.65     60      2.2    2.35      3.96
#     0.85     60      4.6    2.75      4.50   <- 채택
#     1.10     60      8.0    2.98      4.92
#     1.10     90     12.8    3.20      4.78
# half_w 를 키우면 화면 밖(검은 영역)까지 끌어와 전경이 조금 늘지만 낭비가 급증한다.
# 0.85 -> 1.10 은 전경 +0.23%p 에 검은영역 +3.4%p 라 이득이 없다.
HALF_W_NEAR = 0.85
Y_FAR_OFF = 60         # 사다리꼴 위변을 소실점보다 이만큼 아래에 둔다
RECTIFY = 1.0          # 1.0 = 완전 BEV(수직 하방), 낮출수록 비스듬한 시점. build_H 설명 참조


def build_H(vp, img_w, img_h, half_w_near=HALF_W_NEAR, y_far_off=Y_FAR_OFF,
            y_near_frac=0.98, rectify=RECTIFY):
    """소실점 기반 사다리꼴 -> (부분) 직사각형 호모그래피. (H, src, y_far) 반환.

    `rectify` 로 시점 각도를 조절한다.
      1.0 : 목적지가 직사각형 = **완전 BEV**(수직 하방). 차선이 완전한 수직 평행선이 된다
      0.0 : 목적지가 원본 사다리꼴과 같은 비율 = **원근 그대로**(변환 없음에 가까움)
      중간 : 원근을 일부만 펴서 비스듬히 내려다보는 시점

    완전 BEV 는 원거리를 극단적으로 늘려 흐릿해지고 차량 번짐도 심하다. rectify 를 낮추면
    그 왜곡이 줄어드는 대신 차선의 평행성이 떨어진다.
    """
    vx, vy = vp
    y_near = img_h * y_near_frac
    y_far = vy + y_far_off
    xl_near, xr_near = vx - img_w * half_w_near, vx + img_w * half_w_near
    t = (y_far - vy) / (y_near - vy)          # 소실점 기준 선형 보간 비율 = 원본 위변/아래변
    xl_far, xr_far = vx + (xl_near - vx) * t, vx + (xr_near - vx) * t
    src = np.float32([[xl_near, y_near], [xr_near, y_near],
                      [xr_far, y_far], [xl_far, y_far]])

    wb = BEV_W * (1 - MARGIN)                 # 목적지 아래변 폭
    ratio = (1.0 - rectify) * t + rectify     # 목적지 위변/아래변 비율
    wt = wb * ratio
    cx = BEV_W / 2
    dst = np.float32([[cx - wb / 2, BEV_H], [cx + wb / 2, BEV_H],
                      [cx + wt / 2, 0], [cx - wt / 2, 0]])
    return cv2.getPerspectiveTransform(src, dst), src, y_far


def warp_polyline(Hm, pts, y_far, step=2.0):
    """폴리라인을 BEV 좌표로. 소실선 위쪽은 잘라내고, 촘촘히 리샘플해 변환한다.

    끊긴 구간은 별도의 선분 목록으로 돌려준다(중간이 화면 밖으로 나갔다가 들어오는 경우).
    """
    P = np.asarray(pts, float)
    dense = []
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        n = max(2, int(np.hypot(*(b - a)) / step))
        for k in range(n):
            dense.append(a + (b - a) * k / n)
    dense.append(P[-1])
    D = np.array(dense)
    keep = D[:, 1] >= y_far                    # 사다리꼴 위변 아래만
    if keep.sum() < 2:
        return []
    Q = cv2.perspectiveTransform(D.reshape(-1, 1, 2).astype(np.float32), Hm).reshape(-1, 2)
    ok = keep & np.isfinite(Q).all(axis=1)
    ok &= (Q[:, 0] > -BEV_W) & (Q[:, 0] < 2 * BEV_W)
    ok &= (Q[:, 1] > -BEV_H) & (Q[:, 1] < 2 * BEV_H)
    segs, cur = [], []
    for i, f in enumerate(ok):
        if f:
            cur.append(Q[i])
        elif cur:
            if len(cur) >= 2:
                segs.append(np.array(cur))
            cur = []
    if len(cur) >= 2:
        segs.append(np.array(cur))
    return segs


def label_dir(sub):
    p = SRC / f"[라벨]{sub}"
    if not any(f.endswith(".json") for f in os.listdir(p)):
        p = p / next(d for d in os.listdir(p) if (p / d).is_dir())
    return p


def clip_ids(stems):
    """파일명이 연속 정수인 구간 = 영상 클립. 분할 시 누수 차단용 (01단계와 동일)."""
    nums = sorted(int(s) for s in stems)
    out, cid = {}, 0
    for i, n in enumerate(nums):
        if i and n - nums[i - 1] != 1:
            cid += 1
        out[str(n)] = cid
    return out


def render_one(job, Hm, y_far, width, save_img):
    stem, lpath, ipath, oimg, omask = job
    try:
        d = json.load(open(lpath, encoding="utf-8"))
    except Exception as e:
        return (stem, 0, 0, f"json: {e}")

    mask = np.zeros((BEV_H, BEV_W), np.uint8)
    # dotted 먼저, solid 나중 -> 교차 지점에서 solid 가 살아남는다 (01단계와 동일 정책)
    for want in ("dotted", "solid"):
        for a in d.get("annotations", []):
            if a.get("class") != "traffic_lane":
                continue
            t = {x["code"]: x["value"] for x in a.get("attributes", [])}.get("lane_type")
            if t != want:
                continue
            pts = [(p["x"], p["y"]) for p in a.get("data", [])]
            if len(pts) < 2:
                continue
            for seg in warp_polyline(Hm, pts, y_far):
                cv2.polylines(mask, [seg.astype(np.int32)], False,
                              LANE_TYPE_TO_ID[t], width, lineType=cv2.LINE_8)

    if save_img:
        img = cv2.imread(str(ipath))
        if img is None:
            return (stem, 0, 0, "이미지 읽기 실패")
        bev = cv2.warpPerspective(img, Hm, (BEV_W, BEV_H), flags=cv2.INTER_LINEAR)
        oimg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(oimg), bev, [cv2.IMWRITE_JPEG_QUALITY, 92])

    omask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(omask), mask)
    return (stem, int((mask == 1).sum()), int((mask == 2).sum()), "")


def measure_spacing(sub, n_img=250):
    """BEV 에서 인접 차선 간격(픽셀)을 잰다. 선 폭을 정하기 위한 기준값.

    한국 도로 기준 차로 폭 3.0~3.5m, 노면표시 폭 0.10~0.20m -> 비율 약 0.03~0.06.
    """
    vp = VP[sub]
    img_h = 1080 if "1080" in sub else 1200
    Hm, _, y_far = build_H(vp, 1920, img_h)
    ldir = label_dir(sub)
    files = sorted(os.listdir(ldir))
    step = max(1, len(files) // n_img)
    gaps = []
    for f in files[::step]:
        d = json.load(open(ldir / f, encoding="utf-8"))
        xs = []
        for a in d.get("annotations", []):
            if a.get("class") != "traffic_lane":
                continue
            pts = [(p["x"], p["y"]) for p in a.get("data", [])]
            if len(pts) < 2:
                continue
            for seg in warp_polyline(Hm, pts, y_far):
                m = (seg[:, 1] > BEV_H * 0.55) & (seg[:, 1] < BEV_H * 0.95)
                if m.sum() >= 2:
                    xs.append(float(np.median(seg[m, 0])))
        xs = sorted(x for x in xs if 0 <= x <= BEV_W)
        for i in range(len(xs) - 1):
            g = xs[i + 1] - xs[i]
            if 8 < g < BEV_W * 0.6:      # 같은 선의 중복·화면 반대편은 제외
                gaps.append(g)
    return np.array(gaps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DST)
    ap.add_argument("--subsets", nargs="+", default=None)
    # BEV 차선 간격 실측 93px ≈ 차로폭 3.3m -> 약 28 px/m.
    # 노면표시 폭은 일반도로 0.10~0.15m, 고속도로 0.15~0.20m.
    #   0.15m -> 4px,  0.20m -> 6px
    # 4px 는 학습하기에 지나치게 얇아(전경 2.75%) 물리 범위 상단인 0.20m 기준 6px 을 쓴다.
    # 전경 비율이 약 4% 로 올라가 전방 시점(1.62%)의 2.5배가 된다.
    ap.add_argument("--width", type=int, default=6,
                    help="BEV 에서의 선 폭(px). 0 이면 차선 간격 실측으로 자동 결정")
    ap.add_argument("--width-ratio", type=float, default=0.060,
                    help="차선 간격 대비 노면표시 폭 비율 (0.20m / 3.3m ≈ 0.060)")
    ap.add_argument("--measure-only", action="store_true", help="차선 간격만 재고 종료")
    ap.add_argument("--no-images", action="store_true", help="마스크만 생성 (이미지 변환 생략)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    subs = args.subsets or sorted(VP)

    # --- 선 폭 결정 ---
    if args.width:
        width = args.width
        print(f"선 폭 = {width}px (지정)")
    else:
        print("BEV 차선 간격 실측 -> 선 폭 결정")
        allg = []
        for s in subs:
            g = measure_spacing(s)
            if len(g):
                allg.append(g)
                print(f"  {s[:40]:40s} 표본 {len(g):6,d}  중앙 간격 {np.median(g):6.1f}px")
        if not allg:
            print("[!] 간격 측정 실패. --width 로 직접 지정하라")
            return 1
        med = float(np.median(np.concatenate(allg)))
        width = max(2, int(round(med * args.width_ratio)))
        print(f"  전체 중앙 간격 {med:.1f}px x 비율 {args.width_ratio} -> 선 폭 {width}px")
    if args.measure_only:
        return 0

    rows = []
    for sub in subs:
        vp = VP[sub]
        img_h = 1080 if "1080" in sub else 1200
        Hm, src, y_far = build_H(vp, 1920, img_h)
        ldir, idir = label_dir(sub), SRC / f"[원천]{sub}"
        stems = sorted(f[:-5] for f in os.listdir(ldir) if f.endswith(".json"))
        if args.limit:
            stems = stems[:args.limit]
        cid = clip_ids(stems)
        jobs = [(s, ldir / f"{s}.json", idir / f"{s}.jpg",
                 args.out / "images" / sub / f"{s}.jpg",
                 args.out / "masks" / sub / f"{s}.png") for s in stems]
        print(f"[{sub}] {len(jobs):,}장, 클립 {max(cid.values())+1:,}개  VP={vp}  y_far={y_far:.0f}")
        fn = partial(render_one, Hm=Hm, y_far=y_far, width=width, save_img=not args.no_images)
        with Pool(args.workers) as pool:
            for i, (stem, s_px, d_px, err) in enumerate(pool.imap(fn, jobs, chunksize=32), 1):
                if err:
                    print(f"  [!] {stem}: {err}")
                    continue
                rows.append(dict(subset=sub, stem=stem, clip=cid[stem], H=BEV_H, W=BEV_W,
                                 solid_px=s_px, dashed_px=d_px,
                                 img=str(args.out / "images" / sub / f"{stem}.jpg"),
                                 mask=str(args.out / "masks" / sub / f"{stem}.png")))
                if i % 3000 == 0:
                    print(f"  {i:,}/{len(jobs):,}")

    args.out.mkdir(parents=True, exist_ok=True)
    mf = args.out / "manifest.csv"
    with open(mf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tot_s = sum(r["solid_px"] for r in rows)
    tot_d = sum(r["dashed_px"] for r in rows)
    tot_p = len(rows) * BEV_H * BEV_W
    print(f"\n총 {len(rows):,}장 -> {mf}")
    print(f"선 폭 {width}px, BEV {BEV_W}x{BEV_H}")
    print(f"픽셀 비율: background {(tot_p-tot_s-tot_d)/tot_p*100:.3f}%  "
          f"solid {tot_s/tot_p*100:.3f}%  dashed {tot_d/tot_p*100:.3f}%")
    print(f"전경 없는 장: {sum(1 for r in rows if r['solid_px']+r['dashed_px']==0):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
