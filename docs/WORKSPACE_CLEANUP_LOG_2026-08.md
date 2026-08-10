# 워크스페이스 정리 작업 기록 — 2026년 8월

이 문서는 2026년 8월에 `/home/hsjeong/workspace`와 Conda 가상환경을 정리하면서 수행한 삭제, 프로젝트 이전, 경로 재연결 및 검증 내용을 기록한다.

- 작업 기준일: 2026-08-07
- 현재 상태 재확인: 2026-08-10
- 정리 기준 문서: [`WORKSPACE_ORGANIZATION_GUIDE.md`](./WORKSPACE_ORGANIZATION_GUIDE.md)

> 삭제된 폴더와 가상환경은 별도 백업이 없다면 복구할 수 없다. 이 문서는 삭제 대상의 내용을 보존하는 백업이 아니라 작업 이력이다.

## 1. 정리 결과 요약

- 사용하지 않는 연구 프로젝트와 대응 가상환경을 삭제했다.
- SAM3를 `vision-seg` 아래의 프로젝트 단위 구조로 이전했다.
- YOLO-MASTER를 `vision-det` 아래로 이전하고 기존 Conda 환경을 새 경로에 재연결했다.
- YOLO26-MASTER를 `vision-det` 아래로 이전하고 기존 Conda 환경을 새 경로에 재연결했다.
- Yolo11을 `vision-det` 아래로 이전하고 내부 Git 저장소, Conda 환경과 실행 경로를 보존했다.
- 당시 `Yolo11/ultralytics/runs`의 기존 실험 산출물을 정리했다. 이후 프로젝트 이전으로 현재 경로는 `vision-det/Yolo11/ultralytics/runs`이다.
- 프로젝트 구조와 명명 기준은 별도 가이드 문서로 정리했다.

## 2. 삭제한 Conda 가상환경

다음 환경은 사용 프로젝트와 상태를 확인한 뒤 제거했다.

```text
INP_env
Realnet
patchguard_env
CFLOW_env
RepVIT
mmyolo_env
uninet_env
seas_env
cdad_env
PAPER_env
Yolov26_latest_stargate
PaDim_env
efficientad
pbas_env
glass_env
yolov12_env
yolov13_env
lwdetr
Yolov11-SOD_env
YOLO_8.4.50_env
```

### 2026-08-10 기준 남아 있는 환경

```text
base
SOD-PAPER
YOLO-MASTER
YOLO26-MASTER
YOLO_8.4.90_env
Yolov26_env
Yolov26_env_3.10
mqbench_env
paddleocr_gpu_cu126
sam3
segformer_cu111
yolov11_env
```

## 3. 삭제한 프로젝트 및 데이터

다음 경로를 삭제했다.

```text
/home/hsjeong/workspace/PRN
/home/hsjeong/workspace/INP
/home/hsjeong/workspace/MSYolo
/home/hsjeong/workspace/EfficientAD
/home/hsjeong/workspace/Yolov13_enganced
/home/hsjeong/workspace/LW-DETR
/home/hsjeong/workspace/dataset/SEG_COCO
/home/hsjeong/workspace/VIT
/home/hsjeong/workspace/PAPER
/home/hsjeong/workspace/YOLO_8.4.50
```

참고 사항:

- `VIT`는 약 6.9GB였으며, 약 6.8GB의 RepVIT 실험 결과와 체크포인트를 포함하고 있었다.
- `PAPER`에는 삭제 당시 `.gitignore` 하나만 있었다.
- `YOLO_8.4.50`은 약 17MB였고 대응 환경 `YOLO_8.4.50_env`도 함께 제거했다.
- 기존 `Yolo11/ultralytics/runs`는 내용만 정리했으며, 프로젝트 이전 후 위치는 `vision-det/Yolo11/ultralytics/runs`이다.
- 기존 `/home/hsjeong/workspace/Yolo11/runs`의 실험 결과 약 13GB도 프로젝트와 함께 `/home/hsjeong/workspace/vision-det/Yolo11/runs`로 이동했다. 위 `ultralytics/runs`와는 별도의 경로다.

## 4. SAM3 프로젝트 이전

### 현재 위치

```text
/home/hsjeong/workspace/vision-seg/sam3
```

### 구성

```text
vision-seg/sam3/
├── src/        # 원본 Git 저장소와 SAM3 패키지
├── scripts/    # 연구자가 사용하는 실행 스크립트
├── config/     # 프로젝트 설정
├── test/       # 테스트 케이스와 임시 출력
├── results/    # 모델, 샘플, 지표, 그림, 내보낸 결과
├── docs/       # 프로젝트 문서
├── assets/     # 프로젝트 리소스
└── archive/    # 이전 버전 및 백업
```

기존 `/home/hsjeong/workspace/SAM3` 경로는 제거되었으며 Conda 환경 `sam3`는 유지했다.

## 5. YOLO-MASTER 프로젝트 이전

### 경로 변경

```text
이전: /home/hsjeong/workspace/YOLO-MASTER
현재: /home/hsjeong/workspace/vision-det/YOLO-MASTER
```

프로젝트 크기는 이전 당시 약 371MB였다.

### 가상환경 연결

- Conda 환경: `/home/hsjeong/miniconda3/envs/YOLO-MASTER`
- 패키지: `ultralytics 8.3.240`
- 설치 방식: editable install
- 현재 editable 위치: `/home/hsjeong/workspace/vision-det/YOLO-MASTER`

이동 후 다음 명령과 동일한 방식으로 가상환경 연결을 갱신했다.

```bash
/home/hsjeong/miniconda3/envs/YOLO-MASTER/bin/python -m pip install \
  -e /home/hsjeong/workspace/vision-det/YOLO-MASTER \
  --no-deps
```

`TEST_diagnose.py`에 있던 모델 절대경로도 새 프로젝트 위치로 수정했다.

### 수행한 검증

- 새 경로에서 `ultralytics` import 성공
- `yolo` CLI 실행 및 버전 확인 성공
- 로컬 `yolo11n.pt` 모델 로드 성공
- 로컬 샘플 이미지 CPU 추론 성공
- editable 설치 메타데이터가 새 경로를 가리키는 것 확인
- 프로젝트 내부와 workspace의 다른 파일에서 기존 절대경로 참조가 남지 않은 것 확인
- 진단 스크립트가 참조하는 `best.pt` 파일 존재 확인

2026-08-10 재확인 결과 실제 import 위치는 다음과 같다.

```text
/home/hsjeong/workspace/vision-det/YOLO-MASTER/ultralytics/__init__.py
```

### Git 관련 남은 작업

`YOLO-MASTER`의 Git 루트는 프로젝트 폴더가 아니라 `/home/hsjeong/workspace`이다. 이동한 파일을 아직 staging하거나 커밋하지 않았다.

현재 상위 Git에서는 기존 `YOLO-MASTER/ultralytics` 파일들이 삭제된 것으로, `vision-det/YOLO-MASTER`는 새 경로로 표시된다. GitHub에 폴더 이동을 반영하려면 변경 범위를 검토한 다음 별도로 staging 및 commit해야 한다.

## 6. YOLO26-MASTER 프로젝트 이전

### 경로 변경

```text
이전: /home/hsjeong/workspace/YOLO26-MASTER
현재: /home/hsjeong/workspace/vision-det/YOLO26-MASTER
```

- 프로젝트 크기: 이전 당시 약 710MB
- 실험 결과: `runs` 약 680MB, Git 추적 제외
- Conda 환경: `/home/hsjeong/miniconda3/envs/YOLO26-MASTER`
- 패키지: `ultralytics 8.4.50`
- 설치 방식: editable install

이동 전 임시 복사본을 사용해 파일 1,538개와 전체 바이트 크기가 원본과 일치하는지 확인했다. 새 경로에서 editable 설치, import, `yolo` CLI, `yolo26n.pt` 모델 로드와 CPU 추론이 모두 성공한 뒤 실제 이전했다.

이동 후 `make_report.py`의 보고서 출력 절대경로를 새 프로젝트 위치로 수정했으며, 프로젝트 안팎에 기존 절대경로 참조가 남지 않은 것을 확인했다.

GitHub의 `origin/main`에서 기존 `YOLO26-MASTER/ultralytics` 파일 404개가 추적되고 있었다. 현재 상위 Git에는 기존 경로 404개 삭제와 `vision-det/YOLO26-MASTER` 신규 경로로 표시되며, staging이나 commit은 수행하지 않았다.

## 7. Yolo11 프로젝트 이전

> 이 절의 내부 Git 상태는 최초 이전 당시의 기록이다. 이후 `Yolo11/ultralytics`는 아래 11절과 같이 상위 `vision-research` 저장소로 편입했다.

### 경로 변경

```text
이전: /home/hsjeong/workspace/Yolo11
현재: /home/hsjeong/workspace/vision-det/Yolo11
```

- 프로젝트 크기: 이전 당시 약 28GB
- 파일 수: 12,502개
- 주요 용량: `runs` 약 13GB, `wandb` 약 9.8GB
- Conda 환경: `/home/hsjeong/miniconda3/envs/yolov11_env`
- 패키지: `ultralytics 8.3.201`
- 설치 방식: editable install

이동 전 hard-link 임시 복사본에서 editable 설치, import, `yolo` CLI, `yolo11n.pt` 모델 로드와 CPU 추론을 검증했다. 실제 이동 후에도 같은 검증을 통과했다.

활성 내부 스크립트와 설정 8개, `Yolo26`의 외부 코드 참조 3개를 새 경로로 수정했다. 과거 실험의 `args.yaml`, W&B 로그와 run 메타데이터는 당시 실행 기록 보존을 위해 수정하지 않았다. 과거 체크포인트로 학습을 재개할 때는 데이터와 모델 경로를 새 위치로 명시해야 할 수 있다.

### 내부 Git 저장소

`Yolo11` 전체는 상위 workspace Git에서 미추적이지만 다음 두 독립 저장소를 포함한다.

```text
vision-det/Yolo11/ultralytics  # branch SOD-YOLO, 개인 GitHub remote
vision-det/Yolo11/mmcv         # branch main, OpenMMLab remote
```

이동 후 두 저장소의 `.git`, branch, HEAD와 remote가 유지된 것을 확인했다. `ultralytics` 저장소에는 이동 전부터 수정된 캐시 파일 5개와 미추적 캐시 114개가 있었으며 이동 후에도 같은 상태다. `mmcv`는 clean 상태를 유지했다.

### Ultralytics 공용 경로

사용자 공용 Ultralytics 설정은 다음과 같이 갱신했다.

```text
runs_dir:     /tmp/Ultralytics/runs
datasets_dir: /home/hsjeong/workspace/vision-det/Yolo11/datasets
weights_dir:  /home/hsjeong/workspace/vision-det/Yolo11/models
```

`project`를 지정하지 않은 detection 실행의 기본 저장 위치는 다음과 같이 계산된다.

```text
/tmp/Ultralytics/runs/detect/<experiment-name>
```

`/tmp`는 시스템 재부팅이나 임시 파일 정리 정책에 따라 삭제될 수 있으므로 보존할 결과는 프로젝트의 `results` 또는 별도 영구 경로로 옮겨야 한다.

## 8. 현재 주요 프로젝트 배치

```text
workspace/
├── vision-det/
│   ├── YOLO-MASTER/
│   ├── YOLO26-MASTER/
│   └── Yolo11/
├── vision-seg/
│   └── sam3/
├── SOD-PAPER/
├── Yolo26/
├── dataset/
└── docs/
```

## 9. 이후 정리 시 확인 순서

프로젝트를 삭제하거나 옮기기 전에 다음 순서로 확인한다.

1. 프로젝트 크기와 주요 체크포인트 및 결과물 확인
2. Git 루트, tracked/untracked 파일과 remote 확인
3. 연결된 Conda 환경 및 editable 설치 여부 확인
4. 실행 중인 프로세스 확인
5. 프로젝트 안팎의 기존 절대경로 검색
6. 임시 복사본에서 import, CLI 및 최소 추론 검증
7. 실제 이동 후 editable 설치와 절대경로 갱신
8. 기존 경로 제거, 새 경로 import 및 Git 상태 최종 확인

## 10. 보안 메모

정리 과정에서 일부 Git remote 설정에 URL 내 인증정보가 포함된 사례가 확인되었다. 로컬 폴더를 삭제해도 발급된 토큰은 자동 폐기되지 않으므로, URL에 포함됐던 접근 토큰은 GitHub에서 별도로 폐기하거나 교체해야 한다.

## 11. Yolo11 Ultralytics 소스 편입

2026-08-10에 `Yolo11` 전체가 아닌 다음 소스 디렉터리만 상위 `vision-research` 저장소의 관리 대상으로 전환했다.

```text
/home/hsjeong/workspace/vision-det/Yolo11/ultralytics
```

- 상위 저장소 커밋: `744d145 Integrate YOLO11 source under vision-det`
- GitHub 반영 위치: `vision-det/Yolo11/ultralytics/`
- 편입 파일: 소스 및 모델·데이터셋 설정 296개
- 실행 검증: `yolov11_env`에서 `ultralytics` import 경로 정상
- 공용 결과 경로: `/tmp/Ultralytics/runs` 유지

다음 항목은 상위 Git에서 제외했다.

```text
Yolo11의 ultralytics 외 디렉터리
mmcv
runs 및 wandb
모델 가중치 (*.pt, *.pth)
Python 캐시 (__pycache__, *.pyc)
로컬 데이터셋과 기타 실험 산출물
```

기존 독립 저장소 `jeonghs9/secuwatcher-yolo11`의 내부 `.git`은 작업 디렉터리에서 분리했으며, 당시 복구용 사본은 다음 임시 경로에 두었다.

```text
/tmp/Yolo11-ultralytics.git.backup-20260810
```

이 경로는 `/tmp`이므로 영구 보관 위치가 아니다. 기존 GitHub 저장소를 삭제하기 전에는 `git bundle`로 전체 이력을 영구 경로에 백업하거나, 저장소를 삭제하지 않고 Archive 처리해야 한다.

삭제 검토 당시 `secuwatcher-yolo11` 상태는 다음과 같았다.

```text
visibility: private
branches: main, SOD-YOLO
commits: 7 (main 1개, SOD-YOLO에 추가 6개)
tags: 없음
releases: 없음
open issues: 없음
```

현재 `vision-research`에는 최신 소스 스냅샷이 편입됐지만 기존 7개 커밋의 이력 자체는 병합되지 않았다. 따라서 영구 백업 없이 `secuwatcher-yolo11`을 삭제하면 과거 변경 이력을 잃는다.

## 12. Traffic Video Analytics 1단계 분리

2026-08-10에 기존 `OCR-CAR-CATEGORY/UTIL`의 최종 배포 파일을 새 애플리케이션 구조로 복사하고 회귀 검증했다.

```text
원본: /home/hsjeong/workspace/Yolo26/ultralytics/PROJECT/OCR-CAR-CATEGORY/UTIL/visualize_solid_lane_crossing_v5.3.1.py
신규: /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/visualize_solid_lane_crossing_v5.3.1.py
```

- `Yolo26` 코드, 모델, 데이터와 기존 `PROJECT` 원본은 이동하거나 삭제하지 않았다.
- 실제 런타임은 `vision-det/YOLO26-MASTER`가 아니라 `/home/hsjeong/workspace/Yolo26/ultralytics`의 editable `ultralytics 8.4.6`이다.
- 원본과 신규 스크립트의 SHA-256이 동일한 것을 확인했다.
- 두 위치에서 동일 모델·동일 영상의 첫 10프레임을 GPU 1로 실행했다.
- 생성된 MP4와 CSV가 각각 바이트 단위로 동일했다.
- 신규 위치에서 PaddleOCR 모델 3종 초기화와 1프레임 처리를 확인했다.
- 모델과 생성 결과는 `models/`, `outputs/`에 둘 수 있지만 Git에서는 제외한다.

상세 검증 결과와 기존 UTIL 파일 분류는 다음 문서에 기록했다.

```text
applications/traffic-video-analytics/docs/MIGRATION_VERIFICATION_2026-08-10.md
applications/traffic-video-analytics/docs/UTIL_FILE_DISPOSITION.md
```

현재 PaddleOCR 캐시는 여전히 다음 기존 경로를 사용한다.

```text
/home/hsjeong/workspace/Yolo26/ultralytics/UTIL/LPR/License-Plate-Recognition-System
```

따라서 SegFormer 자료, 재현 도구, 과거 프로토타입과 OCR 캐시를 별도로 보존하기 전에는 기존 `OCR-CAR-CATEGORY/UTIL` 또는 관련 상위 경로를 삭제하면 안 된다.
