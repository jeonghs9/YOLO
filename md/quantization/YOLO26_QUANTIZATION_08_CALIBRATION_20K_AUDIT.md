# YOLO26 Calibration 64장 및 BRECQ 20K 재검증

작성일: 2026-07-03
코드 브랜치: `STAR_GATE-MQBench`

## 1. 변경 목적

기존 Advanced PTQ 실험은 train split의 정렬된 앞 16장과 1,000 updates/block을 사용했다.
MQBench 예제의 20,000 updates/block과 비교하면 reconstruction budget이 작고, 정렬된 앞부분
이미지는 전체 배포 분포를 대표하지 못할 가능성이 있었다.

이번 실험에서는 다음을 변경했다.

```text
MQBench calibration: 16 -> 64 images
BRECQ updates: 1,000 -> 20,000 per block
ORT activation calibration: 300 -> 512 images
selection: sorted-first -> deterministic hybrid
validation conf: 0.25 -> 0.001
```

## 2. Calibration 이미지 선정

완벽한 단일 calibration set은 존재하지 않는다. 실제 배포 분포를 따르면서 희소한 activation
패턴도 포함하는 것이 목적이다. 구현한 hybrid sampler는 train split에서 다음 방식으로 선정한다.

```text
50%: seed 기반 전체 random sample
50%: class x object-size coverage가 부족한 이미지를 greedy 선택
object size: normalized box area 기준 small / medium / large
```

64장 결과:

```text
Random 표본의 class 포함 이미지 수
person 60, car 12, motorcycle 17, plate_number 5

Hybrid 표본의 class 포함 이미지 수
person 57, car 37, motorcycle 38, plate_number 33
```

Hybrid 표본에는 네 클래스의 small/medium/large 객체가 모두 포함됐다. Test split은 calibration에
사용하지 않았다.

## 3. 실행 설정

```text
Algorithm: MQBench BRECQ block-wise W8A8
Blocks: 22
Reconstructed Conv: 175
Calibration: hybrid train 64 images
Updates: 20,000/block, total 440,000
ORT calibration: hybrid train 512 images
Detect head: FP32
Input: 1024, batch 1
```

GPU reconstruction 시간은 5,114초, 약 85분이었다.

생성 파일:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/
best_1024_brecq_w8a8_calib64_iter20000_mqbench.pt
best_1024_brecq_w8a8_calib64_iter20000_rounded_fp32.onnx
best_1024_brecq_w8a8_calib64_iter20000_fullops.onnx
```

## 4. 정확도 재검증

모든 비교는 다음 동일 조건으로 다시 측정했다.

```text
data: data_qnsfl_new.yaml
split: test, 6,701 images
imgsz: 1024
batch: 1
conf: 0.001
provider: CPUExecutionProvider
```

```text
                               Precision  Recall  mAP50  mAP50-95
Static Full-op                   0.781     0.627   0.708    0.439
BRECQ 16 images / 1K             0.794     0.617   0.704    0.439
BRECQ 64 hybrid images / 20K     0.808     0.646   0.735    0.468
```

개선 BRECQ와 기존 BRECQ 비교:

```text
mAP50:    +0.031
mAP50-95: +0.029
```

개선 BRECQ와 Static 비교:

```text
mAP50:    +0.027
mAP50-95: +0.029
```

따라서 이전에 BRECQ 개선이 거의 없었던 주원인은 알고리즘 미실행이 아니라 부족한 calibration
대표성과 reconstruction budget이었다.

## 5. 속도

ORT 20 threads, warm-up 10, runs 50:

```text
Static Full-op       95.209 ms, 10.503 FPS
BRECQ 16/1K          94.256 ms, 10.609 FPS
BRECQ 64/20K         96.550 ms, 10.357 FPS
```

세 모델의 속도는 사실상 같다. BRECQ 반복 수와 calibration 수는 weight rounding 정확도를
개선하지만 최종 QOperator 구조나 연산량은 변경하지 않는다.

## 6. 코드 기본값 변경

```text
AdaRound calibration: 64, updates/layer: 20,000, Conv: 175개 전체
BRECQ calibration: 64, updates/block: 20,000
QDrop calibration: 64, updates/block: 20,000
ORT calibration: 512
calibration selection: hybrid
```

AdaRound와 QDrop의 고반복 전체 실행은 각각 장시간이 필요하므로 기존 결과를 덮어쓰지 않고 별도
이름으로 실행해야 한다.
