# v5.3 화면 밖/차량 가림 분리와 점선 gap 복원

> 최종 갱신: 2026-08-04
> 실행 파일: `UTIL/visualize_solid_lane_crossing_v5.3.1.py` (동작 보존 정리판, 10절)
> 결과 재현 기준: `UTIL/visualize_solid_lane_crossing_v5.3.py`
> 이전 버전: `UTIL/visualize_solid_lane_crossing_v5.2.py`
> 차선 모델: `260727_Y26S_V_LANE_seg/weights/best.pt`
> 차량·번호판 모델: `models/3차모델/best.pt`
> 상태: 5개 영상 전체 실행 완료, 육안 검수에서 발화 조건 오탐 확인 →
> `LANE_POSTPROCESS_06_TRIGGER_EVIDENCE_STRENGTH.md`로 이어짐

## 1. 해결하려는 문제

점선인데 실선으로 판정되어 정상 차량이 발화하는 오탐이다. 원인을 v5.2 코드와
001037 영상 300프레임 계측으로 특정했다.

`analyze_signal()`은 점선 gap을 인정할 때 빈칸 **바로 옆** 두 칸이 모두 차선일 것을
요구했다.

```python
and (cleaned[start - 1] == 1)
and (cleaned[end] == 1)
```

고속도로에서 차량은 차선 바로 옆으로 주행하므로, 점선의 빈칸은 하필 차량 bbox가
겹치는 위치에 자주 생긴다. 그러면 신호가 다음 모양이 되어 빈칸이 통째로 폐기된다.

```text
1 1 1 0 0 0 0 0 0 X X 1 1 1     → v5.2: gap 아님 → SOLID
```

증거가 사라지면 `raw_type`은 자동으로 SOLID가 된다. SOLID가 기본값이기 때문이다.

### 계측 결과 (001037, 300프레임, 60px 이상 0 구간 전수)

| 분류 | 개수 | 비율 |
|---|---:|---:|
| X가 인접해 폐기 | 3,927 | 64.1% |
| 유효 gap으로 인정 | 1,639 | 26.8% |
| 배열 끝단이라 제외 (정상) | 561 | 9.2% |

유효 gap 1개당 2.4개가 버려지고 있었다.

## 2. v5.3 변경 내용

변경은 연속성 신호 처리 2곳뿐이다. AST 비교로 97개 함수·클래스 중
`sample_cell_signal`, `analyze_signal` 2개만 변경되었고 `nearest_known_value` 1개가
추가되었음을 확인했다. `parse_cli_args`가 동일하므로 **새 CLI 인자는 없다.**

### 변경 1: 화면 밖과 차량 가림을 분리

v5.2는 두 경우를 모두 `-1`로 기록했다.

```python
SIGNAL_LANE = 1
SIGNAL_EMPTY = 0
SIGNAL_VEHICLE_HIDDEN = -1     # 차량이 가려서 판단 불가
SIGNAL_OUT_OF_FRAME = -2       # 화면 밖 (v5.3 신규)
```

화면 밖은 "증거가 없는 것"이고 차량 가림은 "가려서 모르는 것"이다. v5.2가 도입한
무효표 판정에는 차량 가림만 사용해야 한다.

```python
unknown_ratio = hidden / max(len(cleaned) - out_of_frame, 1)
```

chain ROI는 차선 조각 중심에서 양쪽으로 뻗어나가며 화면 경계를 고려하지 않는다.
cell 500px × 3~5칸이므로 중심에서 최대 ±1,250px까지 확장되고, 화면 가장자리 근처
차선에서는 ROI의 상당 부분이 영상 밖으로 나간다. 분석 해상도는 결과 영상(1920×1080)이
아니라 원본 3840×2160이다.

`sample_cell_signal()`은 slice의 표본이 모두 영상 밖이면 `inside_count == 0`이 되어
해당 slice를 unknown으로 기록한다. v5.2는 이것을 차량 가림과 같은 `-1`로 저장했다.
그 결과 전체 X의 39.7%가 화면 밖이었고 차선 판정의 43.5%가 무효표 처리되고 있었다.

### 변경 2: X를 건너뛰어 바깥쪽 차선을 확인

```python
def nearest_known_value(signal, index, step):
    while 0 <= index < len(signal):
        if signal[index] >= 0:
            return signal[index]
        index += step
    return None
```

빈칸 바깥쪽으로 X를 건너뛰며 처음 만나는 관측값이 양쪽 모두 차선이면 유효 gap으로
인정한다. 배열 양 끝의 0을 제외하는 기존 규칙은 그대로 유지한다.

`0` 구간 자체는 "도로가 보이는데 페인트가 없다"는 확실한 관측이며, 그 옆에 차량이
있다는 사실이 그 관측을 무효화하지 않는다.

### 변경하지 않은 것

- 차량 발화 조건, 접촉 시간, 중앙점 probe, 화면 경계 조건
- 청록 bridge 생성 규칙과 표시/판정 픽셀 일치
- 시간 다수결 파라미터, ROI chain 생성, 렌더링, OCR, CSV schema
- `occupancy` 계산식 (합성 10케이스에서 v5.2와 완전 동일 확인)

## 3. 화면 표시 의미 변경

`--show-lane-rois` 라벨의 `X=` 값은 이제 **차량 가림 비율**만 나타낸다. v5.2까지는
화면 밖 비율이 함께 포함되어 있었다. 같은 장면에서 v5.2보다 낮은 값이 표시된다.

## 4. 합성 검증 (10케이스 전부 통과)

| 케이스 | v5.2 | v5.3 | 비고 |
|---|---|---|---|
| 실선 (0 없음) | SOLID | SOLID | 회귀 없음 |
| 점선 gap 80px, 양쪽 차선 | DOTTED | DOTTED | 회귀 없음 |
| **gap 오른쪽에 차량 가림** | SOLID | **DOTTED** | 목표 수정 |
| **gap 양쪽 모두 차량 가림** | SOLID | **DOTTED** | 목표 수정 |
| **gap 오른쪽이 화면 밖** | SOLID | **DOTTED** | 목표 수정 |
| 가림 너머에 차선 없음 | SOLID | SOLID | 증거 부족 시 회수 안 함 |
| 배열 끝단 0 | SOLID | SOLID | 기존 규칙 유지 |
| 짧은 구멍 20px | SOLID | SOLID | 노이즈 보정 유지 |
| **실선을 차량이 가림 (0 없음)** | SOLID | **SOLID** | **실선 보호 유지** |
| gap 50px (기준 미만) | SOLID | SOLID | 임계값 유지 |

무효표 분모:

```text
신호 = 차선 5 + 차량가림 5 + 화면밖 10
v5.2 X비율 = 0.750  (15/20)
v5.3 X비율 = 0.500  ( 5/10)
```

## 5. 실영상 검증 (001037, 앞 300프레임, 동일 CLI)

| 지표 | v5.2 | v5.3 | 변화 |
|---|---:|---:|---|
| 유효 점선 gap 총 개수 | 1,639 | 2,427 | +48% |
| SKIP-X (무효표) | 43.5% | 25.4% | −18.1%p |
| `temporal_confirmed` | 64.7% | 81.3% | +16.6%p |
| raw DOTTED 그룹 비중 | 42.2% | 56.2% | 역전 |
| stable SOLID 판정 | 25.1% | 20.8% | 감소 |
| stable DOTTED 판정 | 39.5% | 60.5% | 1.53배 |
| 발화 차량 | 4대 | 2대 | 감소 |

폐기되던 gap 3,927개 중 788개(약 20%)가 회수되어 유효 gap이 48% 증가했다. 나머지는
차량이 빈칸을 완전히 덮어 `0`이 하나도 없거나, X 너머에 차선 관측 자체가 없는
경우로 원리적으로 회수가 불가능하다.

## 6. 기각한 대안 — `--continuity-dotted-min-gaps 2`

과교정(실선이 DOTTED로 뒤집힘)을 막기 위해 gap 2개를 요구하는 안을 함께 측정했으나
정반대로 작동해 기각했다.

| 지표 | v5.3 | v5.3 + `dotted-min-gaps 2` |
|---|---:|---:|
| raw DOTTED 그룹 | 1,517 | 630 (−58%) |
| stable SOLID 판정 | 20.8% | 63.2% |
| 발화 차량 | 2대 | 13대 |

한 프레임의 ROI 신호에서 gap 2개가 동시에 보이는 경우가 드물기 때문이다. 나머지는
가려지거나 화면 밖이다. "여러 번 관찰해야 한다"는 요구는 공간(한 프레임 gap 개수)이
아니라 시간(`--lane-vote-*`의 30프레임 투표)에 걸어야 한다.

**`--continuity-dotted-min-gaps`는 기본값 1을 유지한다.**

## 7. 실행 명령

### 검증용 (앞 300프레임, OCR 없음)

```bash
cd /home/hsjeong/workspace/Yolo26/ultralytics/PROJECT/OCR-CAR-CATEGORY

/home/hsjeong/miniconda3/envs/paddleocr_gpu_cu126/bin/python \
  UTIL/visualize_solid_lane_crossing_v5.3.py \
  --det-model "/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt" \
  --lane-model "260727_Y26S_V_LANE_seg/weights/best.pt" \
  --input "/home/hsjeong/workspace/dataset/TEST_VIDEO/19700101_001037_T.mp4" \
  --output "/home/hsjeong/tmp/v53_001037_first300.mp4" \
  --csv "/home/hsjeong/tmp/v53_001037_first300.csv" \
  --solid-contact-hold-seconds 1 \
  --solid-center-radius 9 \
  --border-margin 0 \
  --chain-roi-cell-length 500 \
  --continuity-slice-length 10 \
  --continuity-close-zero-gap-pixels 30 \
  --continuity-dotted-min-gap-pixels 60 \
  --continuity-max-unknown-ratio-for-vote 0.5 \
  --show-lane-rois \
  --no-ocr \
  --device 6 \
  --max-frames 300
```

### 전체 영상 (OCR 포함) — v5.2와 동일한 실험값

001539 는 `--device 1 --paddle-device gpu:1`, 001037 은 `2`, 000816 은 `3` 을 사용한다.

```bash
cd /home/hsjeong/workspace/Yolo26/ultralytics/PROJECT/OCR-CAR-CATEGORY

/home/hsjeong/miniconda3/envs/paddleocr_gpu_cu126/bin/python \
  UTIL/visualize_solid_lane_crossing_v5.3.py \
  --det-model "/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt" \
  --lane-model "260727_Y26S_V_LANE_seg/weights/best.pt" \
  --input "/home/hsjeong/workspace/dataset/TEST_VIDEO/19700101_001037_T.mp4" \
  --output "RESULT_VIDEO/solid_lane_crossing_v5.3_001037T_cell500_ocr.mp4" \
  --csv "RESULT_VIDEO/solid_lane_crossing_v5.3_001037T_cell500_ocr.csv" \
  --solid-contact-hold-seconds 1 \
  --solid-center-radius 9 \
  --border-margin 0 \
  --chain-roi-cell-length 500 \
  --continuity-slice-length 10 \
  --continuity-close-zero-gap-pixels 30 \
  --continuity-dotted-min-gap-pixels 60 \
  --continuity-max-unknown-ratio-for-vote 0.5 \
  --show-lane-rois \
  --device 2 \
  --paddle-device gpu:2
```

## 7.1 전체 영상 실행 결과 (2026-08-04)

5개 영상 전체를 v5.3으로 실행했다. CLI는 v5.2 실행값과 완전히 동일하고 실행 파일명만
다르다. 원본 결과는 `RESULT_VIDEO/`, CRF 28/veryfast 압축본은 `RESULT_VIDEO/용량줄이기/`에
있다.

| 영상 | 프레임 | 원본 크기 | 압축 크기 | 감소율 | 발화 |
|---|---:|---:|---:|---:|---:|
| 000157 | 8,880 | 2,301,334,034 B | 223,415,249 B | 90.3% | 29대 |
| 000411 | 8,942 | 2,170,620,562 B | 243,683,598 B | 88.8% | 11대 |
| 000816 | 8,983 | 2,204,818,918 B | 236,896,000 B | 89.2% | 8대 |
| 001037 | 8,950 | 2,728,816,938 B | 239,771,836 B | 91.2% | 12대 |
| 001539 | 8,961 | 2,068,266,969 B | 230,284,569 B | 88.9% | 2대 |

압축본은 모두 H.264 / 1920×1080 / 30 FPS이며 원본과 프레임 수·재생시간이 일치한다.
분석 자체는 원본 3840×2160에서 수행하고 `--output-scale 0.5`로 저장한 것이다.

### v5.2 대비 발화 변화 (동일 3개 영상)

| 영상 | v5.2 | v5.3 | 유지 | 사라짐 | 신규 |
|---|---:|---:|---:|---|---|
| 001539 | 2 | 2 | 2 | — | — |
| 001037 | 13 | 12 | 11 | `#21@f195`, `#912@f287` | `#2993@f695` |
| 000816 | 6 | 8 | 6 | — | `#25576@f3680`, `#51174@f6802` |
| 합계 | 21 | 22 | 19 | 2 | 3 |

유지된 19대는 **발화 프레임까지 완전히 동일**하다. 분류가 바뀐 5대만 결과가 달라졌고,
발화 로직은 변경되지 않았음이 실데이터로 확인됐다.

신규 3대가 생긴 이유는 `temporal_confirmed`가 64.7% → 81.3%로 올라 **실선도 더 빨리
확정**되기 때문이다. v5.3은 점선을 점선으로 되돌리는 것뿐 아니라 실선 확정도 앞당긴다.

### 번호판 확보율

| 영상 | 발화 | 번호판 확보 |
|---|---:|---:|
| 000157 | 29 | 0 |
| 000411 | 11 | 3 |
| 000816 | 8 | 3 |
| 001037 | 12 | 1 |
| 001539 | 2 | 1 |
| 합계 | 62 | 8 (13%) |

최종 산출물이 위반 차량 번호판 표시인데 확보율이 13%다. 원인 분석은 별도 단계로 다룬다.

## 7.2 `--border-margin` 켬/끔 비교 (000157, 000411)

`--border-margin 0`은 화면 경계에 잘린 차량의 발화 차단을 **해제**한다. 코드에서
`margin > 0`이 거짓이면 뒤 조건을 평가하지 않으므로 `border_clipped`가 항상 False다.
000411 CSV 283,886행 전수에서 `border_clipped=0`임을 확인했다.

경계 제외를 켠 실행(`잘린거제외` 접미사)과 비교하면 다음과 같다.

| 영상 | 제외 OFF | 제외 ON | 걸러진 차량 |
|---|---:|---:|---|
| 000157 | 29 | 27 | `#66580@f4565`, `#130615@f8758` |
| 000411 | 11 | 10 | `#75649@f7718` |

- 신규 발화는 0대다. 경계 제외는 순수하게 걸러내기만 한다.
- 걸러진 3대는 모두 중앙원 픽셀이 8·13·253으로 극단적이다. `#130615`는 원 253픽셀이
  전부 채워져 청록 bridge 의존이 의심되던 차량인데, 동시에 경계 차량이기도 했다.
- `#92784`는 걸러지지 않고 발화가 f=7039 → f=7050으로 11프레임 밀렸다. 경계에 있는
  동안 접촉 시간이 초기화됐다가 화면 안으로 들어온 뒤 다시 누적했기 때문이다.

## 8. 다음 확인 사항

1. ~~`#912` 육안 판정~~ → 사용자 육안 검수 완료. v5.3 결과가 정상으로 확인됐다.
2. ~~세 영상 전체 실행~~ → 5개 영상 전체 실행 완료 (7.1절).
3. `--continuity-max-unknown-ratio-for-vote` 를 조정할지는 아직 결정하지 않았다.
   X 비율의 의미가 바뀌었으므로 v5.2의 0.5와 같은 값이 아니다.
4. 원거리 점선의 gap이 60px 미만이라 SOLID가 되는 문제는 이번 수정 대상이 아니다.
   px/m 정규화가 필요하며 별도 단계로 다룬다.
5. CSV의 차선 관련 17개 열(`stable_lane_type`, `lane_continuity_*` 등)은 v5.2와
   동일하게 여전히 비어 있다. `process_video()`가 `CrossingHit()` 기본값을 그대로
   기록하기 때문이며, 이번 수정 범위 밖이다.
6. **v5.3 육안 검수에서 000157의 발화 다수가 오탐으로 확인됐다.** 차선 분류가 아니라
   차량 발화 조건 쪽 문제이며, 분석과 수정안은
   `LANE_POSTPROCESS_06_TRIGGER_EVIDENCE_STRENGTH.md`에 정리했다.

## 9. 미해결로 남긴 항목

이번 v5.3은 **분류(점선/실선) 문제만** 다루었다. 다음은 의도적으로 손대지 않았다.

- 처음부터 실선을 밟고 들어오는 차량의 미탐 (중앙점 9px 고정 반경)
- 접촉 grace 8프레임과 중앙점 통과 기억의 불일치
- 접촉 시간 충족 후 중앙점 1프레임만으로 발화
- ~~`apply_occlusion_overrides()` 도달 불가 코드~~ → v5.3.1에서 삭제 (10절)

## 10. v5.3.1 — 동작 보존 정리판 (2026-08-04)

`UTIL/visualize_solid_lane_crossing_v5.3.1.py`는 **판정 로직을 하나도 바꾸지 않고**
죽은 코드와 중복만 제거한 파일이다. v5.3은 비교 기준으로 그대로 보존한다.

```text
2,502줄 → 2,369줄 (−133줄)
함수·클래스 98개 → 95개 (83개는 바이트 단위로 무변경)
```

### 10.1 삭제한 항목

| 대상 | 근거 |
|---|---|
| `apply_occlusion_overrides()` | 도달 불가능. 진입 조건이 `lane_type == "DOTTED"` AND `allow_occlusion_override`인데, `stabilize_lane_decisions()`가 `allow_occlusion_override = (stable_type == "SOLID")`, `lane_type = stable_type or raw_type`으로 설정하므로 두 조건은 동시에 성립할 수 없다. 6가지 상태 조합 전수 확인 완료 |
| `clip_lane_axis_to_rect()` | 위 함수에서만 호출 |
| `mean_axial_angle()` | `axial_mean_deg([a, b])`와 완전히 동일. 랜덤 2만 회 비교에서 축각도 차이 0.000e+00 |
| `COLOR_OCCLUDED_SOLID` | 참조 0회 |
| `LaneDecision.occlusion_vehicle_id` | 삭제된 함수에서만 기록 |
| `CONTINUITY_RESULTS` | 기록만 하고 읽는 곳이 없음 |

`OCCLUDED_SOLID`가 한 번도 생성되지 않으므로 관련 분기도 정리했다. 결과값은 같다.

```python
# build_solid_id_map
- if decision.lane_type in {"SOLID", "OCCLUDED_SOLID"} and decision.temporal_confirmed
+ if decision.lane_type == "SOLID" and decision.temporal_confirmed

# lane_display_type — 죽은 분기 제거
- if decision.lane_type == "OCCLUDED_SOLID":
-     return "SOLID"

# 화면 오버레이 — 항상 0이었으므로 리터럴화 (표시 문자열 동일)
- occluded={sum((d.lane_type == 'OCCLUDED_SOLID' for d in decisions))}
+ occluded=0
```

`LANE_POSTPROCESS_05_1_4_CURRENT_TEST.md` 1절의 "`apply_occlusion_overrides()`를 제거하거나
주석 처리하면 안 된다"는 기술은 사실과 다르다. v5.1.3의 `NameError`는 함수가 기여해서가
아니라 본문만 주석 처리하고 호출문을 남겨서 발생한 것이다.

### 10.2 파라미터 정리 (11개)

`del`로 받자마자 폐기하던 파라미터를 시그니처에서 제거했다.

| 함수 | 제거한 파라미터 |
|---|---|
| `stabilize_lane_decisions()` | `polygons, per_car, width, height, args, fps` |
| `evaluate_solid_crossing_trigger()` | `heading_fired, recent_crossing, args` |
| `VehicleTracker.__init__()` | `fps` |
| `VehicleTracker.process()` | `frame_index` |
| `stable_solid_fragment_bridges()` | `_existing_bridges` |

### 10.3 optical-flow 잔재 리터럴화

`process_video()`의 지역변수 `recent_crossing`, `baseline_deg`, `heading_deviation`,
`heading_hold_ratio`, `heading_fired`, `crossing_in_hold`는 항상 같은 값이었다.
변수를 없애고 CSV 열에는 리터럴을 넣어 **출력을 그대로 유지**했다.

```python
"",       # baseline_deg        (optical flow 제거로 항상 빈 값)
"0.000",  # baseline_coherence
"",       # heading_deviation_deg
"0.000",  # heading_hold_ratio
0,        # heading_fired
0,        # crossing_in_hold
```

콘솔 발화 로그의 `heading_dev=n/a`, `heading_hold=0.00`도 리터럴로 바꿨고 출력은 동일하다.

### 10.4 의도적으로 남긴 항목

| 대상 | 이유 |
|---|---|
| `CrossingHit` / `process_video()`의 `hit` | CSV 차선 관련 17개 열을 **채워야 할 자리**다. 지우면 후속 수정이 어려워지므로 주석만 달았다 |
| `classify_connected_lane_groups()`의 `seg_count` 기반 판정 | continuity가 덮어쓰지만 제거 시 동작 변화 위험 |
| `classify_connected_lane_groups()` 말미의 fallback 루프 | 죽은 코드로 보이나 확증이 없어 안전망으로 유지 |

### 10.5 검증

| 항목 | 결과 |
|---|---|
| Python 문법 검사 | 통과 |
| Ruff 린트 | `All checks passed` |
| 함수 단위 AST 비교 | 삭제 3, 추가 0, 수정 12, **무변경 83** |
| 실영상 400프레임 CSV | **SHA-256 완전 일치** |
| 콘솔 발화 로그 | 동일 |

```text
입력  19700101_000157_T.mp4, --max-frames 400, --no-ocr, 나머지 CLI 동일
SHA-256  e43641ebe3c7f45056d83b0905bea62500faf65ed523cebb501a3e7832a89b0a
         v5.3   cmp_5.3.csv
         v5.3.1 cmp_5.3.1.csv
```

CSV가 바이트 동일하므로 **기존 v5.3 전체 영상 결과 5편은 그대로 유효**하며 재실행이
필요 없다. 이후 기준 파일로 v5.3.1을 사용한다.

### 10.6 작업 중 발견

삭제 범위를 줄 번호로 지정하다가 `fragment_projection_gap()`이 함께 삭제됐다. 이 함수는
`stable_solid_fragment_bridges()`에서 여전히 사용한다. Ruff가 `F821 Undefined name`으로
잡아내 복원했다. **줄 번호 기반 블록 삭제 후에는 반드시 린트를 돌려야 한다.**
