# Lane Postprocess 문서 인덱스

> 최종 갱신: 2026-07-31

## 1. 문서 파일명 규칙

차선 segmentation 후처리와 관련해 새로 생성하는 Markdown 문서는 다음 형식을 사용한다.

```text
LANE_POSTPROCESS_<2자리 단계번호>_<UPPER_SNAKE_CASE_주제>.md
```

예시:

```text
LANE_POSTPROCESS_01_INSTANCE_COUNT.md
LANE_POSTPROCESS_02_DIRECTIONAL_ROI.md
LANE_POSTPROCESS_03_SOLID_CROSSING.md
LANE_POSTPROCESS_04_RUN_LENGTH.md
```

규칙:

1. 접두사는 항상 `LANE_POSTPROCESS_`로 고정한다.
2. 단계번호는 `01`부터 시작하고 이전 실험을 대체하지 않는다.
3. 주제는 영문 대문자 `UPPER_SNAKE_CASE`로 작성한다.
4. 각 문서 상단에 최종 갱신일, 대응 Python 파일, 사용 모델을 기록한다.
5. 실험 로직, CLI, 판정 기준, 검증 결과, 한계를 반드시 기록한다.
6. 새 단계가 생기면 이 인덱스의 단계 목록과 현재 채택 상태를 함께 갱신한다.
7. 규칙 도입 전에 만들어진 기존 문서는 링크 파손 방지를 위해 이름을 변경하지 않는다.

## 2. 단계 목록

| 단계 | 상태 | 문서 | Python |
|---|---|---|---|
| 01 | 1차 테스트 | `LANE_POSTPROCESS_01_INSTANCE_COUNT.md` | `UTIL/test_lane_seg_instance_count.py` |
| 02 | 테스트 가능 | `LANE_POSTPROCESS_02_DIRECTIONAL_ROI.md` | `UTIL/test_lane_seg_directional_roi.py` |
| 03 | v3.2.1 지정 시간 접촉+차량 중앙점 통과 조건 구현, 단기 실영상 검증 완료 | `LANE_POSTPROCESS_03_SOLID_CROSSING.md` | `UTIL/visualize_solid_lane_crossing_v3.2.1.py` |
| 04 | v4.1 연속성 판정+1.5초 접촉+차량 중앙점 통과 구현, 단기 실영상 검증 완료 | `LANE_POSTPROCESS_04_CONTINUITY_SIGNAL.md` | `UTIL/visualize_solid_lane_crossing_v4.1.py` |
| 05 | v5.2 X 비율 50% 초과 프레임 무효표 처리, T5 점선 유지 검증 완료 | `LANE_POSTPROCESS_05_NO_OPTICAL_FLOW.md`, `LANE_POSTPROCESS_05_1_2_STANDALONE.md`, `LANE_POSTPROCESS_05_1_4_CURRENT_TEST.md`, `LANE_POSTPROCESS_05_2_UNKNOWN_VOTE_GUARD.md` | `UTIL/visualize_solid_lane_crossing_v5.2.py` |
| 05.3 | v5.3 화면 밖/차량 가림 분리, X 건너뛴 점선 gap 복원, 5개 영상 전체 실행 완료. v5.3.1은 동작 보존 정리판(CSV 바이트 동일 검증 완료) | `LANE_POSTPROCESS_05_3_UNKNOWN_GAP_RECOVERY.md` | `UTIL/visualize_solid_lane_crossing_v5.3.1.py` |
| 06 | v5.3 오탐 원인 분석 완료(접촉 grace·중앙점 1프레임·bridge 관통), 수정 미구현 | `LANE_POSTPROCESS_06_TRIGGER_EVIDENCE_STRENGTH.md` | (v5.4 예정) |

## 3. 공통 구현 규칙

- 모든 차선 후처리 Python 실행 파일은 `tqdm` 프레임 진행률을 표시한다.
- 진행률의 단위는 `frame`으로 통일한다.
- 입력 영상의 전체 프레임 수를 알 수 있으면 `total`에 반영한다.
- `--max-frames` 사용 시 진행률 전체 값도 해당 제한에 맞춘다.
- 새 CLI 인자를 만들거나 실행 명령에 선택 인자를 추가하기 전에 사용자 승인을 받는다.
- 임시 테스트 산출물은 `/home/hsjeong/tmp/`에 저장한다.

## 4. 2026-07-31 현재 인계 기준

- optical flow를 유지하는 비교 기준은 `UTIL/visualize_solid_lane_crossing_v4.1.py`이다.
- **현재 기준 실행 파일은 `UTIL/visualize_solid_lane_crossing_v5.3.1.py`이다.**
  v5.3의 죽은 코드·중복만 제거한 정리판이며 판정 로직은 동일하다. 실영상 400프레임에서
  CSV가 SHA-256 단위로 일치함을 확인했으므로 기존 v5.3 전체 영상 결과 5편은 그대로
  유효하고 재실행이 필요 없다. v5.3과 v5.2는 비교용으로 보존한다.
- v5.3.1에서 삭제한 것: `apply_occlusion_overrides()`(도달 불가),
  `clip_lane_axis_to_rect()`, `mean_axial_angle()`(`axial_mean_deg`와 동일),
  `COLOR_OCCLUDED_SOLID`, `LaneDecision.occlusion_vehicle_id`, `CONTINUITY_RESULTS`,
  그리고 `del`로 즉시 버리던 파라미터 11개. 상세는 05_3 문서 10절.
- `OCCLUDED_SOLID`는 v5.2·v5.3에서도 한 번도 생성되지 않았다. 화면의 `occluded=` 값은
  항상 0이며 v5.3.1은 이를 리터럴 0으로 고정했다.
- v5.3은 점선이 실선으로 잘못 판정되는 오탐만 대상으로 한다. 연속성 신호에서
  화면 밖(`-2`)과 차량 가림(`-1`)을 분리하고, 점선 gap 인정 시 X를 건너뛰어 바깥쪽
  차선을 확인한다. 발화 조건, 중앙점 probe, bridge, CLI 인자는 v5.2와 동일하다.
- v5.3에서 `--show-lane-rois` 라벨의 `X=`는 차량 가림 비율만 의미한다. 화면 밖
  비율이 빠졌으므로 같은 장면에서 v5.2보다 낮게 표시된다.
- `--continuity-dotted-min-gaps`는 1을 유지한다. 2로 올리면 한 프레임에서 gap 2개가
  동시에 보이는 경우가 드물어 대부분 SOLID로 되돌아가고, 300프레임 검증에서 발화가
  2대에서 13대로 늘었다.
- 차량 발화 조건 재실험 파일은 `UTIL/visualize_solid_lane_crossing_v3.2.1.py`이다.
  1.5초 연속 접촉에 차량 중앙점 통과 조건을 추가한 현재 Stage 03 실행 대상이다.
- v4는 v3.2를 대체해 다시 작성한 독립 파이프라인이 아니라, v3.2 전체 로직 위에서
  프레임별 SOLID/DOTTED 원시 판정만 종방향 `1/0/X` 연속성 방식으로 교체한 버전이다.
- 전체 영상 평가에서는 의도적으로 `--max-frames`를 사용하지 않는다.
- 기본 임계값은 합성 신호, 초반 Car #4 구간, 기존 `30G 640.04` 오탐 구간으로 검증했다.
  다음 작업은 전체 영상에서 발화 목록과 육안 정답을 비교해 원근 위치별 임계값을 조정하는
  것이다.
- v3.2 장시간 실행 중 발견된 `heading_deviation=None` 로그 오류는 수정됐다. 방향 정보가
  없는 3초 접촉 발화는 `heading_dev=n/a`로 출력된다.
- v5.0.1의 일반 bbox bridge 전면 차단은 정상 차량까지 미탐시켜 폐기했다. 현재는 확정
  SOLID, polygon/chain ROI 도달, 실제 차선 폭, 중앙점 거리 조건을 모두 통과한 차량당
  bridge 1개만 허용하며 표시 폭은 최대 16px이다.
- 마지막 검증 bridge는 12프레임 유지하고, 접촉 마스크 누락은 최대 8프레임까지 같은
  접촉 구간으로 연결한다.
- Car #4는 기존 v5의 frame 108이 아니라 정확한 1.50초가 되는 frame 110에서 발화한다.
- 콘솔 `hold`는 실제 접촉 초를 출력하며 제거된 방향 조건 값은 `heading_hold`로 분리했다.
- v5.1은 시간 다수결로 확정된 SOLID 조각 2개의 실제 끝점만 청록선으로 연결한다.
  stable DOTTED·미확정·단일 조각에서는 bridge를 만들지 않으며, 차선을 차량 중앙으로
  이동하거나 과거 bridge를 차량 이동에 맞춰 유지하지 않는다.
- v5.1 실제 초반 180프레임에서 bridge 비영점과 Car #4 frame 154 발화를 확인했다.
- v5.1.1은 `--solid-center-radius`를 화면 노란 원과 실제 중앙 판정 원에 함께 적용한다.
  기존 `--center-radius`는 호환상 파싱만 하며 v5.1.1 판정과 표시에 적용하지 않는다.
- v5.1.4는 청록 표시 bridge와 실제 접촉 판정 bridge를 하나의 mask로 만들어 픽셀 범위를
  완전히 일치시킨다. 이전 최대 96px 판정 폭은 제거됐고 현재 최대 폭은 16px이다.
- v5.2는 연속성 신호의 X 비율이 기본 50%를 초과하면 해당 프레임을 SOLID/DOTTED
  어느 쪽에도 투표하지 않고 이전 stable 상태와 투표 이력을 유지한다. 001037 영상의
  T5는 frame 188의 `X=0.748`, frame 192의 `X=0.804`가 무효표 처리되어 DOTTED를
  유지했다.
- v5.2 전체 실행은 001539/GPU 1, 001037/GPU 2, 000816/GPU 3에서 OCR 포함으로
  완료됐다. 공통 CLI 실험값은 `hold=1`, `radius=9`, `border=0`, `cell=500`,
  `slice=10`, `close-gap=30`, `dotted-gap=60`, `X vote limit=0.5`이다.
- 세 원본 결과 영상과 CSV는 `RESULT_VIDEO/`에 있으며, CRF 28/veryfast H.264
  압축본은 `RESULT_VIDEO/용량줄이기/`에 있다. 압축 후 크기는 각각 001539 약 220MiB,
  001037 약 229MiB, 000816 약 227MiB이다.
- 현재 비교 시험값은 `hold=1`, `radius=9`, `border=0`, `cell=500`이다. 코드 기본값
  `hold=1.5`, `radius=15`, `border=8`, `cell=300`을 변경한 것은 아니며 CLI에서만
  명시한다.
- 0816 앞 10초, 300프레임 cell 500 디버그에서 Car #4가 frame 123에 발화했다.
- 전체 v5.1 결과에서 #50844는 7.233초, #57480은 6.600초 접촉했으나 두 차량 모두
  중앙 probe의 SOLID 픽셀이 0이라 발화하지 않았다.
- 다음 필수 작업은 세 v5.2 압축 영상과 CSV를 함께 보며 정상 발화, 미탐, 오탐,
  `SKIP-X` 발생 구간을 육안 검수하는 것이다.
- v5.3은 5개 영상(000157/000411/000816/001037/001539) 전체 실행이 끝났고 압축본도
  `RESULT_VIDEO/용량줄이기/`에 있다. 전체 발화는 62대, 번호판 확보는 8대(13%)다.
- v5.3 육안 검수 결과 실선/점선 분류는 정상 동작으로 확인됐다. 다만 000157에서
  차량 발화 조건 쪽 오탐이 다수 확인됐다. 원인 4가지와 수정안은
  `LANE_POSTPROCESS_06_TRIGGER_EVIDENCE_STRENGTH.md`에 있다.
- 발화 조건은 "80% 영역 접촉 AND 중앙점 접촉"이 **같은 프레임에 동시 성립**해야 한다.
  000157 발화 29건 전수에서 확인했다. 구조는 정상이고 문제는 증거 강도다.
- `--border-margin 0`은 경계 차량 발화 차단을 **해제**하는 값이다. `margin > 0`이
  거짓이면 뒤 조건을 평가하지 않으므로 `border_clipped`가 항상 False가 된다.
  켜려면 `--border-margin 8`(코드 기본값)을 쓴다. 000157/000411 비교에서 켰을 때
  3대가 걸러졌고 신규 발화는 0대였다.
- CSV의 차선 관련 17개 열(`lane_type`, `stable_lane_type`, `lane_continuity_*` 등)은
  `process_video()`가 빈 `CrossingHit()`을 기록해 v5.2와 v5.3 모두 전 행이 빈 값이다.
  차선 상태 검수는 현재 CSV가 아니라 화면 ROI 라벨로만 가능하다.
