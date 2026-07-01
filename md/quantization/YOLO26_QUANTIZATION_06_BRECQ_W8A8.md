# YOLO26 BRECQ W8A8 적용 결과

작성일: 2026-07-01
코드 브랜치: `STAR_GATE-MQBench`

## 1. 알고리즘과 구현 범위

AdaRound와 BRECQ는 모두 MQBench의 Advanced PTQ에 포함된 알고리즘이다. 이번 작업은 별도의
유사 알고리즘을 만든 것이 아니라 다음 MQBench 구현을 직접 사용했다.

```text
MQBench AdaRoundFakeQuantize
MQBench LossFunction
learned_hard_sigmoid weight rounding
rounding regularization + block output reconstruction loss
```

MQBench 기본 `ptq_reconstruction(pattern="block")`은 YOLO26의 함수형 residual `Add`를 module처럼
조회하면서 `KeyError: add`가 발생한다. 따라서 양자화 및 최적화 알고리즘은 유지하고 block 추출만
YOLO26 구조에 맞게 교체했다.

```text
기본 MQBench: generic FX block extractor
이번 adapter: model.model.<index> semantic stage extractor
```

각 semantic block에는 내부 Conv, activation, residual Add, Concat이 함께 들어간다. 여러 Conv의
반올림 파라미터를 동시에 학습해 block의 최종 FP32 출력을 복원한다. 이것이 Conv 하나씩 복원하는
AdaRound와의 핵심 차이다.

## 2. 실험 설정

```text
Algorithm: MQBench BRECQ block-wise PTQ
Precision: W8A8
Weight: QInt8, per-channel symmetric
Activation: QUInt8, per-tensor asymmetric
Blocks: 22
Reconstructed Conv: 175
Calibration: train split 16 images
Optimization: 1,000 updates per block, total 22,000 updates
Optimizer: Adam + cosine scheduler
Warm-up: 0.2
Rounding regularization weight: 0.01
Detect head: reconstruction 및 INT8 대상에서 제외, FP32 유지
Input: 1024x1024, batch 1
```

MQBench 예제는 block당 20,000회도 사용하지만 계산량이 매우 크다. 이번 1차 baseline은 AdaRound의
총 21,600 updates와 유사한 총 22,000 updates로 맞춰 알고리즘 차이를 비교했다.

재구성은 Tesla V100 GPU 5에서 약 371초가 걸렸다.

## 3. 생성 파일

MQBench BRECQ 연구 체크포인트:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/
best_1024_brecq_w8a8_mqbench.pt
```

이 파일은 일반 Ultralytics checkpoint가 아니며 실제 CPU INT8 커널을 실행하지 않는다.

Hard-rounded FP32 중간 ONNX:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/
best_1024_brecq_rounded_fp32.onnx
```

실제 CPU W8A8 배포 모델:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/
best_1024_brecq_w8a8_fullops.onnx
```

배포 모델 크기는 `21,892,700 bytes`, 약 `20.88 MiB`다.

## 4. CPU INT8 그래프

BRECQ로 결정된 weight를 hard rounding한 뒤, AdaRound 실험과 동일하게 train 300장으로 ORT
MinMax activation calibration을 수행했다.

```text
QLinearConv       175
QLinearAdd         40
QLinearMul        123
QLinearSigmoid    110
QLinearConcat      28
FP32 Detect Conv   32
QuantizeLinear      4
DequantizeLinear    8
```

최종 파일은 fake INT8이 아니라 ORT CPU가 실제 QOperator INT8 연산을 실행하는 모델이다. BRECQ는
weight rounding 정확도를 담당하고 ORT calibration은 최종 CPU 런타임의 activation range를 정한다.

## 5. 정확도 검증

```text
data: data_qnsfl_new.yaml
split: test, 6,701 images / 34,587 instances
imgsz: 1024
batch: 1
conf: 0.25
provider: ONNX Runtime CPUExecutionProvider
```

결과:

```text
                         Precision  Recall  mAP50   mAP50-95
FP32 baseline 기록          -         -     0.7643   0.5057
MinMax W8A8 Full-op       0.7807    0.6267  0.7076   0.4390
AdaRound W8A8 Full-op     0.794     0.613   0.737    0.508
BRECQ W8A8 Full-op        0.804     0.608   0.735    0.510
```

비교:

```text
BRECQ vs MinMax:   mAP50 +0.0274, mAP50-95 +0.0710
BRECQ vs AdaRound: mAP50 -0.0020, mAP50-95 +0.0020
BRECQ vs FP32:     mAP50 -0.0293, mAP50-95 +0.0043
```

표시 정밀도 기준 BRECQ와 AdaRound의 정확도는 거의 같다. BRECQ는 mAP50-95가 소폭 높고 mAP50은
소폭 낮다. 0.002 차이는 seed와 calibration sample 변화에 따라 바뀔 수 있는 작은 차이이므로,
BRECQ가 명확히 우수하다고 결론 내리기보다는 동급으로 보는 것이 타당하다.

## 6. CPU 속도

검증 당시 서버 load average가 68 이상이어서 절대 속도가 평상시보다 낮았다. 동일 시점에 네 모델을
연속 측정해 상대 비교만 유효하게 만들었다.

```text
조건: ORT 20 intra-op threads, batch 1, 1024, warm-up 10, runs 50
서버: Intel Xeon Gold 6148, load average 68.88 -> 68.16

                         Mean latency  Median    P95       FPS
FP32 baseline             198.445 ms   194.466   211.567   5.039
MinMax W8A8 Full-op       146.956 ms   143.362   173.800   6.805
AdaRound W8A8 Full-op     143.320 ms   140.306   163.231   6.977
BRECQ W8A8 Full-op        142.894 ms   141.345   156.740   6.998
```

BRECQ와 AdaRound의 속도 차이는 0.3% 이내로 사실상 동일하다. 두 모델의 QOperator 구조가 같기
때문이다. 정상 부하에서 이전 AdaRound 측정은 약 10.01 FPS였으므로 BRECQ도 같은 수준이 예상되지만,
이번에는 혼잡 상태의 실제 측정치 `6.998 FPS`만 공식 기록으로 사용한다.

전체 validation은 높은 서버 부하 때문에 inference `202.1 ms/image`로 측정됐다. 이 값은 모델
자체 차이가 아닌 CPU 경쟁 영향을 크게 받았으므로 알고리즘 속도 비교에는 사용하지 않는다.

## 7. 재현 명령

BRECQ block reconstruction:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/hsjeong/workspace/Yolo26/ultralytics \
/home/hsjeong/miniconda3/envs/mqbench_env/bin/python \
  -m ultralytics.utils.quantization.run_brecq_ptq \
  --model /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt \
  --data-yaml /home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml \
  --output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/best_1024_brecq_w8a8_mqbench.pt \
  --split train --imgsz 1024 --calib-images 16 --max-count 1000 --device cuda:0
```

Hard-rounded ONNX:

```bash
PYTHONPATH=/home/hsjeong/workspace/Yolo26/ultralytics \
/home/hsjeong/miniconda3/envs/mqbench_env/bin/python \
  -m ultralytics.utils.quantization.export_adaround_onnx \
  --checkpoint /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/best_1024_brecq_w8a8_mqbench.pt \
  --output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/best_1024_brecq_rounded_fp32.onnx
```

실제 CPU W8A8 Full-op:

```bash
/home/hsjeong/miniconda3/envs/mqbench_env/bin/python \
  -m utils.quantization.export_ort_cpu_int8 \
  --model-input /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/best_1024_brecq_rounded_fp32.onnx \
  --model-output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/best_1024_brecq_w8a8_fullops.onnx \
  --data-yaml /home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml \
  --split train --imgsz 1024 --max-calib-images 300 \
  --op-types Conv Add Mul Sigmoid Clip Concat MaxPool Resize Split Reshape Transpose
```

## 8. 결론

MQBench BRECQ W8A8은 YOLO26에서 정상 적용됐고 실제 CPU INT8 모델까지 변환됐다. 기본 MinMax
W8A8의 정확도 손실은 대부분 회복했지만, 현재 1,000 updates/block 설정에서는 AdaRound보다
뚜렷하게 우수하지 않았다. 실무 기준으로는 더 단순하고 재구성이 빠른 AdaRound가 현재 우선이며,
BRECQ의 추가 이득을 확인하려면 calibration image와 block당 update 수를 늘린 2차 실험이 필요하다.
