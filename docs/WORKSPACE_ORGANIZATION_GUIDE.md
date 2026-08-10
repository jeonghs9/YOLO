# 연구 워크스페이스 정리 가이드

이 문서는 `/home/hsjeong/workspace`의 연구 코드, 데이터셋, 실험 결과와 문서를 일관된 방식으로 관리하기 위한 기준이다.

핵심 원칙은 **연구 분야 아래에 프로젝트를 두고, 프로젝트 하나가 원본 코드부터 최종 결과까지 완결된 구조를 갖게 하는 것**이다.

> 이 문서는 정리 기준만 정의한다. 기존 폴더를 자동으로 이동하거나 이름을 변경하지 않는다.

## 1. 권장 최상위 구조

```text
workspace/
├── vision-det/             # 객체 검출, 소형 객체 검출
├── vision-seg/             # 세그멘테이션, SAM, 차선 분할
├── vision-ocr/             # 번호판 및 문자 인식
├── model-opt/              # 양자화, 경량화, 모델 압축
├── datasets/               # 여러 프로젝트가 공유하는 데이터셋
├── docs/                   # 워크스페이스 공용 문서
└── PROJECTS.md             # 전체 연구 프로젝트 인덱스
```

### 디렉터리 역할

| 디렉터리 | 역할 | 프로젝트 예시 |
|---|---|---|
| `vision-det/` | detection 계열 연구 | `small-object-detection`, `yolo11`, `yolo26` |
| `vision-seg/` | segmentation 계열 연구 | `sam3`, `road-lane-segmentation` |
| `vision-ocr/` | OCR 계열 연구 | `vehicle-plate-ocr` |
| `model-opt/` | 모델 최적화 연구 | `mqbench-qat`, `ptq`, `lora` |
| `datasets/` | 공유 데이터셋 원본과 전처리본 | `aitod`, `visdrone`, `aihub` |
| `docs/` | 프로젝트에 속하지 않는 공용 문서 | 운영 가이드, 연구 인덱스 |

최상위 분야명은 필요할 때만 추가한다. 프로젝트 하나를 위해 분야 폴더를 계속 늘리지 않는다.

## 2. 프로젝트 이름

프로젝트 폴더는 시간이 지나도 연구 목적을 기억할 수 있도록 설명적으로 작성한다. 프로젝트는 가장 가까운 연구 분야 아래에 둔다.

```text
vision-det/
└── small-object-detection/

vision-seg/
├── sam3/
└── road-lane-segmentation/

vision-ocr/
└── vehicle-plate-ocr/

model-opt/
└── yolo-quantization/
```

### 이름 규칙

- 영문 소문자와 하이픈(`-`)을 사용한다.
- 기반 기술 자체를 관리하는 프로젝트는 `sam3`, `yolo26`처럼 기술명을 사용해도 된다.
- 특정 연구 문제는 `small-object-detection`, `vehicle-plate-ocr`처럼 목적을 이름에 담는다.
- `test`, `new`, `final`, `project1`처럼 맥락이 없는 이름은 사용하지 않는다.
- 하나의 폴더는 하나의 연구 질문 또는 명확한 목표를 나타낸다.

좋은 예:

```text
small-object-detection
vehicle-detection-and-plate-ocr
road-lane-segmentation
```

피해야 할 예:

```text
YOLO_PROJECT
final_test
new_model_2
3차모델
```

## 3. 프로젝트 내부 구조

모든 연구 프로젝트는 같은 골격을 사용한다. 프로젝트 폴더만 열어도 원본 코드, 사용자 스크립트, 시험 출력과 최종 결과의 위치를 바로 알 수 있어야 한다.

```text
vision-seg/sam3/
├── README.md
├── src/                # Git 원본 저장소
├── scripts/            # 직접 작성한 실행·변환·평가 스크립트
├── config/             # 실행 설정과 데이터 YAML
├── docs/               # 프로젝트 설명, 실험 기록, 사용법
├── assets/             # 예제 이미지·영상 등 작은 리소스
├── test/               # 테스트 코드·샘플·임시 실행 출력
├── results/            # 채택한 최종 결과와 모델
└── archive/            # 이전 버전과 폐기하지 않을 과거 작업
```

| 경로 | 내용 |
|---|---|
| `README.md` | 프로젝트 목적, 설치법, 주요 스크립트, 대표 결과 |
| `src/` | clone한 원본 Git 저장소. `.git`도 이 안에 둔다 |
| `scripts/` | 학습, 평가, 변환, 시각화 실행 스크립트 |
| `config/` | 모델·학습·평가 설정과 데이터 YAML |
| `docs/` | 요구사항, 실험 설명, 분석과 사용 문서 |
| `assets/` | README용 이미지, 작은 샘플과 시각 자료 |
| `test/` | 테스트 코드, 작은 샘플과 자유롭게 지울 수 있는 실행 결과 |
| `results/` | 검토 후 채택한 모델, 표, 그림과 배포 파일 |
| `archive/` | 이전 구현, 교체된 스크립트와 참고용 백업 |

### 3.1 `src` 원본과 사용자 코드를 분리한다

`src/`는 외부 Git 저장소의 원형을 최대한 유지한다. 직접 사용하는 파이썬 파일과 실행 셸은 프로젝트 루트의 `scripts/`에 둔다.

```text
vision-seg/sam3/
├── src/
│   ├── .git/
│   ├── sam3/
│   ├── pyproject.toml
│   └── README.md
└── scripts/
    ├── batch-multiclass-yolo.py
    └── batch-solid-white-lane.py
```

원본 코드 자체를 수정해야 한다면 수정 내역을 Git commit으로 남긴다. 원본 저장소 안에 임시 스크립트나 결과 이미지를 추가하지 않는다.

### 3.2 `test`와 `results`를 엄격히 구분한다

```text
test/
├── cases/                  # 테스트 코드와 작은 fixture
└── output/                 # 재생성 가능, 필요하면 전체 삭제 가능
    ├── images/
    ├── videos/
    ├── predictions/
    └── logs/

results/                    # 검토 후 보존하기로 결정한 결과
├── models/
├── exports/
├── metrics/
├── figures/
└── samples/
```

- `test/cases/`는 보존하고 `test/output/`만 `.gitignore`에 등록한다.
- `results/`에는 대표 결과만 승격한다. 모든 epoch와 중간 출력은 넣지 않는다.
- 최종 모델은 `results/models/`에 두고 모델 카드 또는 간단한 설명을 함께 남긴다.
- 대용량 원본 데이터셋은 `assets/`나 `test/`에 복사하지 않는다.

### 3.3 SAM3 적용 예시

현재 `SAM3/sam3` 저장소를 기준으로 하면 목표 구조는 다음과 같다.

```text
vision-seg/sam3/
├── README.md
├── src/                            # 현재 SAM3/sam3 Git 저장소
├── scripts/
│   ├── batch-multiclass-yolo.py   # 현재 sam3_batch_multiclass_yolo.py
│   └── batch-solid-white-lane.py  # 현재 sam3_batch_solid_white_lane.py
├── config/
├── docs/
├── assets/
├── test/
│   ├── cases/
│   └── output/
│       ├── images/
│       └── videos/
├── results/
│   ├── models/
│   ├── figures/
│   └── samples/
└── archive/
```

`src/` 내부에 이미 존재하는 upstream `assets/`, `examples/`, `scripts/`는 원본 저장소 구성으로 유지한다. 프로젝트 루트의 `scripts/`와 `assets/`는 사용자가 직접 관리하는 파일만 담는다.

## 4. 실험 결과 이름

반복 실험은 `test/output/`에서 수행하고, 보존할 실험만 `results/experiments/`로 옮긴다. 이름은 `순번-핵심변경점` 형식을 사용한다.

```text
results/experiments/
├── 001-yolo11-baseline/
├── 002-piou/
├── 003-piou-c3k2/
├── 004-star-backbone/
└── 005-star-lora/
```

날짜, GPU 번호, seed, 전체 데이터셋 이름과 긴 하이퍼파라미터는 폴더명에 넣지 않는다. 해당 정보는 `config.yaml`에 기록한다.

### 실험 내부 구조

```text
003-piou-c3k2/
├── config.yaml
├── metrics.csv
├── weights/
│   ├── best.pt
│   └── last.pt
├── figures/
└── logs/
```

학습과 검증은 별도 실험으로 나누지 않는다.

```text
# 피해야 할 형태
003-piou-train/
004-piou-val/

# 권장 형태
003-piou/
├── train/
└── eval/
```

## 5. `config.yaml` 최소 기록 항목

```yaml
project: vision-det/small-object-detection
experiment: 003-piou-c3k2
status: complete

model: yolo11s
task: detection
dataset: visdrone
image_size: 1024
seed: 42

method:
  loss: piou
  backbone: c3k2

environment: SOD-PAPER
code_revision: "<git commit>"
started_at: 2026-08-06
notes: "PIoU와 C3k2 조합 비교"
```

## 6. 프로젝트 `README.md` 템플릿

```markdown
# Small Object Detection

AI-TOD, VisDrone, DOTA에서 소형 객체 검출 성능을 개선하는 연구.

## 목표

- APs 개선
- 작은 객체의 localization 오류 감소

## 구성

- Base model: YOLO11s
- Methods: PIoU, C3k2, STAR backbone
- Datasets: AI-TOD, VisDrone, DOTA
- Metrics: APs, mAP50-95
- Environment: SOD-PAPER

## 현재 상태

- Status: active
- Best run: `005-star-lora`
- Next: DOTA 교차 데이터셋 평가
```

## 7. 루트 `PROJECTS.md` 템플릿

`PROJECTS.md`는 전체 연구의 목차 역할을 한다.

```markdown
# Research Projects

| 경로 | 목적 | 상태 | 대표 결과 | 환경 |
|---|---|---|---|---|
| vision-det/small-object-detection | 소형 객체 AP 개선 | active | 005-star-lora | SOD-PAPER |
| vision-ocr/vehicle-plate-ocr | 차량 검출 및 번호판 OCR | active | 018-flow-association | paddleocr_gpu_cu126 |
| vision-seg/road-lane-segmentation | 도로 차선 분할 | paused | 007-segformer | segformer_cu111 |
| model-opt/yolo-quantization | PTQ/QAT 및 경량화 | active | 012-mqbench-qat | mqbench_env |
```

상태값은 아래 네 가지로 통일한다.

```text
idea       아직 시작하지 않음
active     진행 중
paused     일시 중단
complete   종료
```

## 8. 데이터셋 구조

공유 데이터셋은 프로젝트 내부에 복사하지 않고 `datasets/`에서 한 번만 관리한다.

```text
datasets/
├── aitod/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── visdrone/
│   ├── raw/
│   ├── processed/
│   └── splits/
└── aihub/
    ├── vehicle/
    ├── plate/
    └── lane/
```

- `raw/`: 수정하지 않는 원본
- `processed/`: 포맷 변환, 정제, 타일링 결과
- `splits/`: train/val/test 목록과 YAML

프로젝트에서는 데이터셋을 복사하지 않고 설정 파일의 경로 또는 심볼릭 링크로 참조한다.

### 8.1 분할 결과에는 반드시 `vis/`를 만든다

**train/val/test 분할을 만들 때마다 `vis/`를 함께 생성한다. 선택이 아니라 필수다.**

```text
splits/<분할이름>/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── vis/                        # test 무작위 30장, 라벨을 원본 위에 시각화
└── SPLIT.md                    # 분할 근거와 통계
```

| 항목 | 규칙 |
|---|---|
| 대상 | **test** split |
| 장수 | **30장** |
| 선택 | 무작위 (**seed 고정** — 재실행 시 같은 장이 나와야 비교가 된다) |
| 파일명 | 원본 stem 그대로 (원본과 대조 가능하게) |

**패널 구성**은 전처리 유무로 갈린다.

```text
전처리 없음 : 이미지 | 라벨 오버레이                         (2분할)
전처리 있음 : 원본(변환 전) | 변환 후 | 라벨 오버레이         (3분할)
```

BEV(IPM), 타일링, 크롭처럼 **모습이 크게 바뀌는 전처리**를 했다면 3분할이라야 변환이
맞았는지 판단할 수 있다. 변환 후 이미지만 보면 "원본에 있던 차선이 제대로 넘어왔는지"를
알 수 없다. 각 패널 위에 제목과 클래스별 픽셀 수를 적는다.

원본과 변환본의 **종횡비가 다르면 폭이 아니라 높이를 맞춰** 나란히 놓는다
(예: 원본 1920x1200 종횡비 1.6 vs BEV 512x768 종횡비 0.67 — 폭을 맞추면 한쪽이 뭉개진다).

**왜 필수인가.** 라벨 변환은 조용히 틀린다. 좌표계가 뒤집히거나(예: `image_size`가
`[height, width]` 순서), 클래스 ID가 밀리거나, 폴리라인이 잘못 래스터화돼도 파일은 정상적으로
만들어지고 학습도 돌아간다. 몇 시간을 태운 뒤 성능이 이상해서야 알게 된다.
**30장만 눈으로 보면 30초 만에 잡힌다.**

라벨이 폴리라인·폴리곤처럼 **원본과 다른 형태**라면 변환 결과를 보여줘야 한다.
GT 패널에 보이는 것이 원본 라벨이 아니라 **변환된 마스크**임을 문서에 명시한다.

**선이 얇은 데이터(차선, 균열, 전선 등)는 축소 시 사라진다.** 시각화용으로 팽창시켰다면
"라벨 N px 굵게 표시"처럼 이미지 안에 반드시 적는다. 안 그러면 실제 라벨 두께로 오해한다.

### 8.2 `SPLIT.md` 최소 기록 항목

```markdown
# split: subset-holdout

- 생성: 2026-08-07  /  스크립트: scripts/make_split.py
- 원본: datasets/aihub/lane/raw
- 분할 단위: **클립**(연속 프레임 묶음) — 프레임 단위 분할은 누수

| split | 클립 | 장수 | 전경% |
|---|---:|---:|---:|
| train | 1,419 | 18,094 | 3.73 |
| val | 352 | 4,461 | 3.65 |
| test | 594 | 6,423 | 3.58 |

## 분할 근거
- test = 특정 서브셋 통째로 홀드아웃 (다른 주행이라 누수 없음)
- 서브셋 내부 클립 간 ID 거리 118~453 vs 서브셋 간 7,768~43,260

## 검증
- split 간 공유 클립 0
- 이미지/라벨 stem 불일치 0
- 라벨 고유값 {0,1,2}, dtype uint8
- vis/ 30장 육안 확인 완료
```

### 8.3 영상에서 뽑은 프레임은 반드시 클립 단위로 나눈다

주행 영상·CCTV처럼 **연속 프레임**을 담은 데이터셋은 인접 프레임이 사실상 같은 장면이다
(실측: 인접 프레임의 화소 평균 절대차가 무작위 쌍의 1/3). 프레임 단위로 무작위 분할하면
같은 장면이 train과 test에 동시에 들어가 **성능이 부풀려진다.**

- 파일명이 연속 정수면 그 구간을 하나의 클립으로 보고 **클립째 배정**한다
- 가능하면 한 단계 더 나아가 **촬영 세션(폴더) 단위로 test를 홀드아웃**한다
- 분할 후 **split 간 공유 클립이 0인지 assert**로 확인한다

## 9. 모델 보관 규칙

모든 학습 체크포인트는 임시 실행 중에는 `test/output/`에 둔다. 실제 사용할 최종 모델만 해당 프로젝트의 `results/models/`로 승격한다.

```text
vision-det/small-object-detection/results/models/
├── sod-piou-v1.pt
└── sod-star-v2.engine
```

최종 모델 파일명은 다음 형식을 권장한다.

```text
<목적>-<방법 또는 모델>-v<버전>.<확장자>
```

## 10. 현재 워크스페이스의 배치와 예상 매핑

아래 표에서 `이전 완료` 항목은 실제 경로 이동과 실행 검증이 끝난 상태다. `검토` 항목은 정리 방향을 설명하기 위한 제안이며 실제 이동 전 경로 의존성을 점검해야 한다.

| 상태 | 기존 또는 현재 경로 | 권장 또는 적용 위치 |
|---|---|---|
| 이전 완료 | `SAM3` | `vision-seg/sam3` |
| 이전 완료 | `SAM3/sam3` Git 저장소 | `vision-seg/sam3/src` |
| 이전 완료 | `YOLO-MASTER` | `vision-det/YOLO-MASTER` |
| 이전 완료 | `YOLO26-MASTER` | `vision-det/YOLO26-MASTER` |
| 이전 완료 | `Yolo11` | `vision-det/Yolo11` |
| 검토 | `SOD-PAPER` | `vision-det/small-object-detection` |
| 검토 | `Yolo26/ultralytics/PROJECT/OCR-CAR-CATEGORY` | `vision-ocr/vehicle-plate-ocr` |
| 검토 | `Yolo26/ultralytics/SWEEP` | 관련 프로젝트의 `test/sweeps` |
| 검토 | `Yolo26/ultralytics/UTIL` | 관련 프로젝트의 `scripts` |
| 검토 | `vision-det/YOLO26-MASTER`, `Yolo26` | `vision-det/yolo26` 구조로 통합 |
| 검토 | `vision-det/YOLO-MASTER`, `vision-det/Yolo11` | 역할을 확인한 뒤 `vision-det/yolo11` 구조로 통합 |
| 검토 | `dataset` | `datasets` |
| 이전 완료 | `md` | `docs` |

### 10.1 Ultralytics 공용 실행 결과 경로

현재 사용자 공용 Ultralytics 설정의 기본 실행 결과 경로는 다음과 같다.

```text
/tmp/Ultralytics/runs
```

`project`를 생략하면 task별로 다음과 같은 경로가 사용된다.

```text
/tmp/Ultralytics/runs/detect/<experiment-name>
/tmp/Ultralytics/runs/segment/<experiment-name>
/tmp/Ultralytics/runs/classify/<experiment-name>
```

이 경로는 자유롭게 지울 수 있는 임시 실행 결과용이다. `/tmp`는 재부팅이나 시스템 정리로 삭제될 수 있으므로 채택한 체크포인트, 지표, 표와 그림은 프로젝트의 `results/`로 승격한다. 보존이 필요한 학습에서는 코드나 CLI의 `project`를 영구 경로로 명시한다.

## 11. 적용 순서

기존 경로를 한 번에 변경하면 데이터 YAML, 학습 스크립트, 체크포인트와 로그에 저장된 절대경로가 깨질 수 있다. 다음 순서로 점진적으로 적용한다.

1. `PROJECTS.md`를 만들고 현재 연구를 분야별로 등록한다.
2. 새 프로젝트부터 표준 골격을 적용한다.
3. `SAM3`처럼 작고 구조가 분명한 프로젝트 하나로 먼저 시험한다.
4. Git 원본을 `src/`, 사용자 코드를 `scripts/`로 분리한다.
5. 기존 경로에는 임시 심볼릭 링크를 두어 호환성을 유지한다.
6. 데이터셋의 원본과 복사본을 구분하고 마지막에 이동한다.
7. 이동 후 학습·평가 스크립트의 smoke test를 실행한다.
8. 문제가 없을 때 기존 호환 링크를 제거한다.

## 12. 최종 체크리스트

- 프로젝트 이름만 보고 연구 목적을 이해할 수 있는가?
- 각 프로젝트에 최신 `README.md`가 있는가?
- Git 원본 저장소가 프로젝트의 `src/`에 분리되어 있는가?
- 직접 작성한 실행 코드가 `scripts/`에 모여 있는가?
- 임시 출력은 `test/output/`, 채택한 결과는 `results/`에 있는가?
- 보존한 실험이 `001-...`, `002-...` 순서로 정리되어 있는가?
- `final`, `new`, `test2`, `train10` 같은 이름이 남아 있지 않은가?
- 데이터셋 원본이 여러 프로젝트에 복사되어 있지 않은가?
- 최종 모델과 중간 체크포인트가 구분되어 있는가?
- 종료된 연구가 `archive/`에 분리되어 있는가?
- `PROJECTS.md`에서 현재 연구 상태를 한눈에 볼 수 있는가?
- **모든 분할에 `vis/` 30장과 `SPLIT.md`가 있는가? (8.1~8.2절)**
- **영상 프레임 데이터를 클립 단위로 나눴는가? (8.3절)**
