# YOLO26 AdaRound W8A8 적용 결과

작성일: 2026-07-01
코드 브랜치: `STAR_GATE-MQBench`

## 1. 목적

기존 ORT MinMax W8A8 Full-op 모델은 CPU 속도는 개선됐지만 정확도 손실이 컸다.
AdaRound로 각 Conv weight의 INT8 반올림 방향을 최적화한 뒤, 동일한 CPU QOperator
Full-op 구조로 변환해 정확도 회복 여부를 확인했다.

## 2. 적용 방식

```text
알고리즘: MQBench AdaRound, layer-wise PTQ
Weight: 8-bit, per-channel symmetric
Activation simulation: 8-bit, per-tensor asymmetric
재구성 대상: torch.fx 그래프에서 추적된 Conv 108개
Calibration: train split 16장
최적화: Conv당 200 updates, Adam + cosine scheduler
Detect: 재구성 및 CPU INT8 대상에서 제외, FP32 유지
입력: 1024x1024, batch 1
```

MQBench 기본 advanced PTQ extractor는 YOLO26의 함수형 residual `Add`에서 실패했다.
따라서 MQBench의 `AdaRoundFakeQuantize`와 `LossFunction`을 그대로 사용하면서 forward hook으로
각 Conv의 입력과 FP32 목표 출력을 수집하는 layer-wise 재구성 runner를 구현했다.

AdaRound는 weight를 단순히 가장 가까운 INT8 값으로 반올림하지 않는다. 각 weight를 아래 또는
위 눈금 중 어느 쪽으로 보낼지 Conv 출력 오차가 작아지는 방향으로 학습한다. Activation 범위는
이후 ORT MinMax calibration으로 정한다.

## 3. 생성 파일

MQBench 정확도 시뮬레이션 체크포인트:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/
best_1024_adaround_w8a8_mqbench.pt
```

이 파일은 일반 Ultralytics `best.pt`가 아니다. AdaRound state와 실험 metadata를 저장한 연구용
체크포인트이며, 실제 CPU INT8 커널을 실행하지 않는다.

AdaRound hard-rounded FP32 중간 ONNX:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/
best_1024_adaround_rounded_fp32.onnx
```

실제 CPU W8A8 배포 모델:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/
best_1024_adaround_w8a8_fullops.onnx
```

파일 크기:

```text
FP32 baseline ONNX          72.53 MiB
AdaRound hard-rounded ONNX  72.50 MiB
AdaRound W8A8 Full-op       20.96 MiB
```

## 4. 실제 INT8 그래프

ORT Static PTQ를 사용해 AdaRound weight를 실제 CPU QOperator로 내렸다. Activation은 train split
300장으로 MinMax calibration했다.

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

따라서 이 모델은 fake INT8이 아니다. Backbone과 Neck의 주요 연산은 ORT CPU의 정수 연산자로
실행되고 Detect head는 FP32로 남는다.

## 5. 정확도

검증 조건:

```text
data: data_qnsfl_new.yaml
split: test, 6,701 images / 34,587 instances
imgsz: 1024
batch: 1
conf: 0.25
device: CPUExecutionProvider
```

결과:

```text
                         Precision  Recall  mAP50   mAP50-95
FP32 baseline 기록          -         -     0.7643   0.5057
MinMax W8A8 Full-op       0.7807    0.6267  0.7076   0.4390
AdaRound W8A8 Full-op     0.794     0.613   0.737    0.508
```

AdaRound Full-op은 기존 MinMax Full-op 대비 `mAP50 +0.0294`, `mAP50-95 +0.0690`을 기록했다.
표시 정밀도 내에서는 FP32 baseline의 mAP50-95도 유지했다. mAP50은 FP32보다 0.0273 낮다.

MQBench fake-quant graph 자체 검증에서는 `mAP50 0.7500`, `mAP50-95 0.4811`이었다. 이 값은
`conf=0.001`로 측정했으므로 위 `conf=0.25` 배포 표와 직접 비교하지 않는다.

## 6. CPU 속도

서버 CPU `Intel Xeon Gold 6148`, ORT 20 intra-op threads, batch 1, 1024 입력, warm-up 10회,
측정 50회 조건이다.

```text
                         Mean latency  Median    P95       FPS
FP32 baseline             159.149 ms   154.014   192.847   6.283
MinMax W8A8 Full-op       100.275 ms    98.621   114.382   9.973
AdaRound W8A8 Full-op      99.864 ms    99.701   105.308  10.014
```

AdaRound W8A8은 FP32보다 약 `1.59x` 빠르다. 기존 MinMax W8A8과 속도가 거의 같은 이유는 두
모델이 동일한 W8A8 QOperator 구조를 실행하기 때문이다. AdaRound는 속도를 더 높이는 알고리즘이
아니라 같은 INT8 속도에서 weight 반올림 오차를 줄이는 정확도 회복 알고리즘이다.

전체 Ultralytics validation 로그의 AdaRound inference는 `110.9 ms/image`, 약 `9.02 FPS`였다.
전처리 2.5 ms와 후처리 0.2 ms가 별도로 측정됐다.

## 7. 재현 명령

AdaRound 재구성:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/hsjeong/workspace/Yolo26/ultralytics \
/home/hsjeong/miniconda3/envs/mqbench_env/bin/python \
  -m ultralytics.utils.quantization.run_adaround_ptq \
  --model /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt \
  --data-yaml /home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml \
  --output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/best_1024_adaround_w8a8_mqbench.pt \
  --split train --imgsz 1024 --calib-images 16 --max-count 200 \
  --device cuda:0 --validate --val-split test --val-batch 1 --val-workers 4
```

Hard-rounded ONNX 생성:

```bash
PYTHONPATH=/home/hsjeong/workspace/Yolo26/ultralytics \
/home/hsjeong/miniconda3/envs/mqbench_env/bin/python \
  -m ultralytics.utils.quantization.export_adaround_onnx \
  --checkpoint /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/best_1024_adaround_w8a8_mqbench.pt \
  --output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/best_1024_adaround_rounded_fp32.onnx
```

실제 CPU W8A8 Full-op 생성:

```bash
/home/hsjeong/miniconda3/envs/mqbench_env/bin/python \
  -m utils.quantization.export_ort_cpu_int8 \
  --model-input /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/best_1024_adaround_rounded_fp32.onnx \
  --model-output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/best_1024_adaround_w8a8_fullops.onnx \
  --data-yaml /home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml \
  --split train --imgsz 1024 --max-calib-images 300 \
  --op-types Conv Add Mul Sigmoid Clip Concat MaxPool Resize Split Reshape Transpose
```

## 8. 결론

이번 결과에서는 AdaRound가 목적에 맞게 작동했다. 실제 CPU INT8 속도는 기존 Full-op W8A8과
동일하게 유지하면서, 기본 MinMax weight rounding에서 발생한 정확도 손실을 크게 회복했다.
다음 단계는 고객사 CPU에서 동일 모델을 검증하고, BRECQ/QDrop이 추가 정확도 개선을 만드는지
같은 조건으로 비교하는 것이다.
