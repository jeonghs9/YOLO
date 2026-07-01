# YOLO26 CPU INT8 양자화 작업 기록

작성일: 2026-07-01
작업 브랜치: `STAR_GATE-MQBench`
관련 커밋: `467e0bb` (`Support full-op CPU INT8 export`)

## 1. 작업 목적

실무에 사용 중인 YOLO26 StarGate 객체 검출 모델의 정확도는 충분하지만 CPU 추론 속도를 높여야
했다. 단순히 파일 크기만 줄이는 것이 아니라, CPU가 실제 INT8 연산을 실행하도록 만드는 것을
목표로 했다.

원본 모델:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt
```

검증 데이터:

```text
/home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml
test: 6,701 images, 34,587 instances
input: 1024x1024, batch 1
```

## 2. 용어 구분

- **양자화 알고리즘**: FP32 숫자를 INT8로 바꾸는 범위와 반올림 방법을 결정한다.
- **MQBench**: LSQ, BRECQ, QDrop 등의 알고리즘을 적용하고 검증하는 프레임워크다.
- **ONNX**: 모델 저장 형식이다. 그 자체가 양자화 알고리즘은 아니다.
- **ONNX Runtime(ORT)**: ONNX 모델을 CPU/GPU에서 실행하는 런타임이다.
- **QOperator**: `QLinearConv`처럼 실제 정수 연산자를 포함하는 ONNX 표현 방식이다.
- **Fake Quantization**: INT8 오차를 흉내 내지만 실제 계산은 FP32인 정확도 시뮬레이션이다.

## 3. MQBench 적용 결과

MQBench가 YOLO26을 `torch.fx`로 추적할 수 있도록 다음 처리를 했다.

1. 단순한 `forward(self, x)` wrapper 적용
2. C3k2 계열을 `forward_split`으로 변경
3. 공유 activation module을 deepcopy
4. Detect, Attention 계열과 StarGate 일부를 leaf module로 처리

결과:

```text
FakeQuantize: 453개
Weight: 8-bit per-channel symmetric
Activation: 8-bit per-tensor
Calibration: 256 images
```

정확도:

```text
FP32                     mAP50-95 0.5057, mAP50 0.7643
MQBench calibration PTQ  mAP50-95 0.5029, mAP50 0.7660
수동 LSQ QAT             mAP50-95 0.4993, mAP50 0.7596
```

주의: MQBench 실험은 `LearnableFakeQuantize`를 사용했지만 calibration 후 scale을 추가 학습하지
않았다. 따라서 엄밀한 LSQ 학습 결과라기보다 LSQ fake quantizer를 사용한 calibration PTQ다.

MQBench QDQ ONNX는 ORT CPU에서 `QLinearConv`로 변환되지 않았다.

```text
FP32 ONNX    159.7 ms, 6.26 FPS
MQBench QDQ  332.4 ms, 3.01 FPS
QLinearConv  0개
```

FP32 Conv는 그대로 있고 Quantize/Dequantize만 추가되어 느려졌다.

## 4. Conv-only 실제 CPU INT8

MQBench QDQ 대신 ONNX Runtime의 Static PTQ를 사용해 실제 QOperator 모델을 만들었다.

적용 방식:

```text
Algorithm: ONNX Runtime MinMax Static PTQ
Calibration: train images 300장
Weight: QInt8, per-channel symmetric
Activation: QUInt8, per-tensor asymmetric
Precision: W8A8
Detect head /model.31/: FP32 유지
```

Conv-only 모델:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/ONNX/
best_1024_cpu_qoperator_int8.onnx
```

그래프:

```text
QLinearConv       175
FP32 Detect Conv   32
QuantizeLinear    130
DequantizeLinear  151
```

Conv 자체는 빨라졌지만 Conv 사이의 Sigmoid, Mul, Add, Concat 등이 FP32여서 INT8과 FP32 변환이
반복됐다. Windows i7-13700K 결과에서도 변환 비용 때문에 전체 모델은 느렸다.

```text
FP32, 16 threads:       76.736 ms, 13.032 FPS
Conv-only INT8, 8 threads: 106.610 ms, 9.380 FPS
```

## 5. Full-op 실제 CPU INT8

Conv 사이의 형 변환을 줄이기 위해 Backbone과 Neck의 연산을 가능한 한 INT8 상태로 연결하고,
Detect head만 FP32로 유지했다.

양자화 대상:

```text
Conv, Add, Mul, Sigmoid, Clip, Concat,
MaxPool, Resize, Split, Reshape, Transpose
```

생성 모델:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/ONNX/
best_1024_cpu_qoperator_int8_fullops.onnx
```

그래프:

```text
QLinearConv       175
QLinearAdd         40
QLinearMul        123
QLinearSigmoid    110
QLinearConcat      28
FP32 Detect Conv   32
QuantizeLinear     10
DequantizeLinear   11
```

Conv-only 모델과 비교하면 Q/DQ 변환 경계가 `130/151`에서 `10/11`로 크게 줄었다.

재현 명령:

```bash
python -m utils.quantization.export_ort_cpu_int8 \
  --model-input /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best_1024.onnx \
  --model-output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/ONNX/best_1024_cpu_qoperator_int8_fullops.onnx \
  --data-yaml /home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml \
  --split train --imgsz 1024 --max-calib-images 300 \
  --op-types Conv Add Mul Sigmoid Clip Concat MaxPool Resize Split Reshape Transpose
```

## 6. Full-op INT8 정확도

`data_qnsfl_new.yaml`의 전체 test split으로 검증했다.

```text
Precision  0.7807
Recall     0.6267
mAP50      0.7076
mAP50-95   0.4390
```

FP32 기록과 비교:

```text
             mAP50    mAP50-95
FP32         0.7643   0.5057
Full INT8    0.7076   0.4390
차이        -0.0567  -0.0667
```

속도는 개선됐지만 mAP50-95가 절대값 기준 0.0667 감소해 현재 상태는 배포용으로 부적합하다.

## 7. 서버 CPU 속도

서버 CPU:

```text
Intel Xeon Gold 6148 x2
40 cores / 80 threads
AVX-512 지원, VNNI 미지원
```

20 ORT threads, 네트워크 전용 50회 벤치마크:

```text
FP32            167.736 ms, 5.962 FPS
Conv-only INT8  249.083 ms, 4.015 FPS
Full-op INT8    103.142 ms, 9.695 FPS
```

Full-op INT8은 FP32보다 약 1.63배, Conv-only INT8보다 약 2.41배 빨랐다.

전체 test validation에서 측정된 Full-op INT8 속도:

```text
Preprocess   3.1 ms/image
Inference  113.3 ms/image, 8.83 FPS
Postprocess  0.2 ms/image
전체 처리    약 8.58 FPS
```

## 8. GPU 결과 해석

이 모델은 CPU용 QOperator INT8 모델이다. CUDAExecutionProvider로 열어도 많은 QLinear 연산이
CPU로 fallback되고 GPU와 CPU 사이에 127개의 Memcpy 노드가 추가됐다.

```text
CUDA + CPU 혼합 실행: 257.2 ms, 3.89 FPS
```

이 값은 순수 GPU 속도가 아니므로 GPU 성능 지표로 사용하면 안 된다. GPU INT8 성능을 측정하려면
GPU에서 지원되는 QDQ 또는 TensorRT 배포 모델이 별도로 필요하다.

## 9. 다른 PTQ/QAT 알고리즘과의 차이

- **현재 MinMax PTQ**: 관찰된 최소/최대 범위를 256개 INT8 눈금으로 단순 분할한다. 빠르고 구현이
  단순하지만 이상치에 민감하다.
- **Percentile/Entropy PTQ**: 이상치 영향을 줄이거나 정보 손실이 작은 범위를 찾는다. MinMax보다
  정확도가 좋아질 가능성이 있다.
- **AdaRound**: 각 weight를 위/아래 중 어느 방향으로 반올림할지 최적화한다.
- **BRECQ**: YOLO 블록 단위로 FP32 출력과 양자화 출력을 비슷하게 복원한다.
- **QDrop**: 블록 복원 중 activation 양자화를 일부 무작위로 제외해 저비트 activation 오차를 줄인다.
- **LSQ QAT**: 추가 학습으로 INT8 눈금 간격 자체를 최적화한다.

AdaRound, BRECQ, QDrop, LSQ는 주로 정확도를 회복하는 방법이다. 실제 CPU 속도를 얻으려면 최종
모델이 현재 Full-op 모델처럼 실제 QLinear 연산자로 변환되어야 한다.

## 10. 현재 결론과 다음 단계

확인된 사실:

1. Conv-only INT8은 Q/DQ 오버헤드 때문에 FP32보다 느릴 수 있다.
2. Backbone/Neck의 Full-op INT8은 Q/DQ를 줄여 서버 CPU에서 실제 속도 향상을 만들었다.
3. 기본 MinMax PTQ로 Full-op 양자화를 하면 정확도 손실이 크다.
4. CPU QOperator 모델은 GPU 성능 평가에 적합하지 않다.

다음 실험 우선순위:

1. Percentile/Entropy calibration으로 Full-op INT8 정확도 비교
2. AdaRound baseline 적용 완료: `YOLO26_QUANTIZATION_05_ADAROUND_W8A8.md`
3. BRECQ block reconstruction 적용 완료: `YOLO26_QUANTIZATION_06_BRECQ_W8A8.md`
4. QDrop 적용 완료: `YOLO26_QUANTIZATION_07_QDROP_W8A8.md`
5. 각 방법의 mAP50, mAP50-95, CPU FPS를 동일 조건으로 비교
6. 고객사 CPU에서 최종 검증

## 11. 관련 코드와 기록

통합 양자화 문서:

```text
/home/hsjeong/workspace/md/quantization/
YOLO26_QUANTIZATION_01_OVERVIEW.md
YOLO26_QUANTIZATION_02_MQBENCH_CPU_INT8.md
YOLO26_QUANTIZATION_03_WINDOWS_HANDOFF.md
YOLO26_QUANTIZATION_04_WINDOWS_RESULTS.md
YOLO26_QUANTIZATION_05_ADAROUND_W8A8.md
YOLO26_QUANTIZATION_06_BRECQ_W8A8.md
YOLO26_QUANTIZATION_07_QDROP_W8A8.md
```

변환 코드:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/ultralytics/utils/quantization/
export_ort_cpu_int8.py
mqbench_adapter.py
```

초기 MQBench 분석:

```text
/home/hsjeong/workspace/Yolo26/tasks/mqbench_analysis.md
```
