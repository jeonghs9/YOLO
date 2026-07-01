# YOLO26 QDrop W8A8 적용 결과

작성일: 2026-07-01
코드 브랜치: `STAR_GATE-MQBench`

## 1. W8A8 의미

`W8A8`은 `Weight 8-bit + Activation 8-bit`의 약어다.

```text
W8: Conv/Linear weight를 8-bit 정수로 저장하고 연산
A8: layer 사이의 feature map을 8-bit 정수로 변환하고 연산
```

FP32는 값 하나에 32 bit를 사용하지만 INT8은 8 bit를 사용하므로 이론적으로 weight 메모리는
약 1/4이 된다. 실제 속도 향상은 CPU가 `QLinearConv` 같은 INT8 커널을 실행할 때 발생한다.

이번 모델은 Backbone과 Neck이 W8A8이고 정확도 보호를 위해 Detect head의 Conv 32개는 FP32다.
따라서 모델 전체가 100% INT8인 것은 아니다.

## 2. QDrop이란

QDrop은 MQBench Advanced PTQ에 포함된 공식 알고리즘이다. BRECQ처럼 block 단위로 weight rounding을
최적화하지만, 재구성 중 activation quantization을 항상 적용하지 않고 확률적으로 적용한다.

```text
drop_prob=0.5
약 50% activation element: INT8 오차 적용
나머지 약 50%: FP32 값 유지
```

Calibration sample 16장에 activation quantization을 항상 적용하면 그 16장에 과적합될 수 있다.
QDrop은 매 iteration마다 다른 FP32/INT8 혼합 입력을 만들어 더 다양한 양자화 오차 방향을 학습한다.
최종 추론에서는 무작위 drop을 사용하지 않고 activation을 정상적으로 전부 양자화한다.

사용한 MQBench 구성요소:

```text
AdaRoundFakeQuantize: weight 반올림 학습
QDropFakeQuantize: activation 무작위 quantization과 scale 학습
LossFunction: block 출력 복원 + rounding regularization
```

MQBench 기본 block extractor는 YOLO26의 함수형 residual `Add`에서 실패하므로 BRECQ와 동일하게
`model.model.<index>` semantic block adapter를 사용했다. QDrop 알고리즘 자체를 새로 만든 것은 아니다.

## 3. 실험 설정

```text
Algorithm: MQBench QDrop block-wise PTQ
Precision: W8A8
Blocks: 22
Reconstructed Conv: 175
Calibration: train split 16 images
Optimization: 1,000 updates/block, total 22,000 updates
Activation quantization probability: 0.5
Activation scale learning rate: 4e-5
Warm-up: 0.2
Rounding regularization weight: 0.01
Detect head: FP32
Input: 1024x1024, batch 1
```

Tesla V100 GPU 5에서 재구성에 약 507.5초가 걸렸다.

## 4. 생성 파일

MQBench 연구 체크포인트:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/
best_1024_qdrop_w8a8_mqbench.pt
```

Hard-rounded FP32 중간 ONNX:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/
best_1024_qdrop_rounded_fp32.onnx
```

실제 CPU W8A8 배포 모델:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/
best_1024_qdrop_w8a8_fullops.onnx
```

배포 모델 크기는 `21,892,701 bytes`, 약 `20.88 MiB`다.

## 5. 실제 CPU INT8 그래프

QDrop으로 학습한 weight rounding 결과를 저장한 후, train 300장으로 ORT MinMax activation
calibration을 수행했다. QDrop에서 학습한 activation perturbation은 weight rounding에 반영되고,
최종 CPU QOperator의 activation range는 ORT가 배포 형식에 맞게 다시 계산한다.

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

## 6. 정확도

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
QDrop W8A8 Full-op        0.791     0.620   0.738    0.509
```

세 advanced PTQ 모델의 정확도는 사실상 동급이다. QDrop은 mAP50이 가장 높지만 차이는
AdaRound 대비 +0.001, BRECQ 대비 +0.003뿐이다. mAP50-95는 AdaRound보다 +0.001, BRECQ보다
-0.001이다. 이 정도 차이로 특정 알고리즘이 명확히 우수하다고 판단하기는 어렵다.

## 7. CPU 속도

```text
조건: ORT 20 intra-op threads, batch 1, 1024, warm-up 10, runs 50
서버: Intel Xeon Gold 6148, load average 35.16 -> 29.76

                         Mean latency  Median    P95       FPS
FP32 baseline             142.361 ms   138.966   154.495   7.024
MinMax W8A8 Full-op        97.921 ms    94.354   121.825  10.212
AdaRound W8A8 Full-op      97.464 ms    93.807   111.357  10.260
BRECQ W8A8 Full-op         95.116 ms    93.128   103.479  10.513
QDrop W8A8 Full-op         98.078 ms    94.186   120.483  10.196
```

QDrop은 같은 시점의 FP32보다 약 1.45배 빠르다. Advanced PTQ 세 모델의 속도 차이는 약 3% 이내다.
모두 동일한 QOperator 구조를 실행하므로 알고리즘은 주로 정확도에 영향을 주고 속도에는 거의
영향을 주지 않는다.

전체 validation 로그의 QDrop inference는 `114.4 ms/image`, 약 `8.74 FPS`였다.

## 8. 재현 명령

QDrop 재구성:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/hsjeong/workspace/Yolo26/ultralytics \
/home/hsjeong/miniconda3/envs/mqbench_env/bin/python \
  -m ultralytics.utils.quantization.run_qdrop_ptq \
  --model /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt \
  --data-yaml /home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml \
  --output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/best_1024_qdrop_w8a8_mqbench.pt \
  --split train --imgsz 1024 --calib-images 16 --max-count 1000 \
  --drop-prob 0.5 --scale-lr 4e-5 --device cuda:0
```

Hard-rounded ONNX:

```bash
PYTHONPATH=/home/hsjeong/workspace/Yolo26/ultralytics \
/home/hsjeong/miniconda3/envs/mqbench_env/bin/python \
  -m ultralytics.utils.quantization.export_adaround_onnx \
  --checkpoint /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/best_1024_qdrop_w8a8_mqbench.pt \
  --output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/best_1024_qdrop_rounded_fp32.onnx
```

실제 CPU W8A8 Full-op:

```bash
/home/hsjeong/miniconda3/envs/mqbench_env/bin/python \
  -m utils.quantization.export_ort_cpu_int8 \
  --model-input /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/best_1024_qdrop_rounded_fp32.onnx \
  --model-output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/best_1024_qdrop_w8a8_fullops.onnx \
  --data-yaml /home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml \
  --split train --imgsz 1024 --max-calib-images 300 \
  --op-types Conv Add Mul Sigmoid Clip Concat MaxPool Resize Split Reshape Transpose
```

## 9. 결론

MQBench QDrop W8A8 적용과 실제 CPU INT8 배포 검증은 성공했다. 기본 MinMax W8A8의 정확도 손실을
회복했지만 현재 설정에서는 AdaRound 및 BRECQ보다 명확한 우위는 없었다. 실무 기준으로는 가장
단순한 AdaRound가 우선 후보이고, 세 방법을 구분하려면 calibration image 수와 update 수를 늘려
반복 seed 실험을 해야 한다.
