# Windows i7-13700K CPU INT8 결과 인수인계

## 목적

Windows의 Intel Core i7-13700K에서 YOLO26 StarGate FP32 ONNX와 ORT QOperator CPU INT8 ONNX의
네트워크 추론 속도를 같은 조건으로 비교했다. 이 문서는 Linux 서버 Codex가 후속 양자화 작업을 이어가기 위한
인수인계 문서다.

## 사용 모델

Windows 경로:

```text
C:\workspace\models\best_1024.onnx
C:\workspace\models\best_1024_cpu_qoperator_int8.onnx
```

파일 정보:

```text
FP32  76,049,022 bytes
SHA256 11df8ccabffa41292b4f5a6023b7468eddda2bdabbe46ff8b82123939b92eef0

INT8  21,925,621 bytes
SHA256 393be321685a2cf5dbd5c92a2f2ccc92532ec97d74f3501986ed9659afafaf46
```

두 모델의 입력은 `(1, 3, 1024, 1024)`, 출력은 `(1, 300, 6)`이다.

INT8 그래프 연산자:

```text
QLinearConv      175
Conv              32
QuantizeLinear   130
DequantizeLinear 151
```

FP32 Conv 32개는 모두 `/model.31/` Detect head에 있다.

## 벤치마크 조건

- CPU: Intel Core i7-13700K
- OS: Windows 10.0.19045
- ONNX Runtime: 1.27.0
- Execution Provider: `CPUExecutionProvider`만 사용
- batch 1, 1024x1024
- 동일한 seed 0 FP32 입력 tensor 사용
- session 생성과 전처리 시간 제외, `session.run()`만 측정
- ORT sequential execution, inter-op thread 1
- Windows 고성능 전원 모드에서 측정 후 원래 균형 조정 모드로 복원
- 각 조건 warmup 20회, 본 측정 100회

실행 스크립트:

```text
utils/quantization/benchmark_ort_cpu.py
```

명령:

```powershell
.\.venv\Scripts\python.exe utils\quantization\benchmark_ort_cpu.py `
  --fp32 C:\workspace\models\best_1024.onnx `
  --int8 C:\workspace\models\best_1024_cpu_qoperator_int8.onnx `
  --threads 8 16 24 --warmup 20 --runs 100
```

## 원시 속도 결과

```text
platform: Windows-10-10.0.19045-SP0
processor: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
onnxruntime: 1.27.0
providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
input: shape=(1, 3, 1024, 1024), dtype=float32, seed=0
warmup: 20, measured runs: 100
FP32: C:\workspace\models\best_1024.onnx (76049022 bytes, sha256=11df8ccabffa41292b4f5a6023b7468eddda2bdabbe46ff8b82123939b92eef0)
INT8: C:\workspace\models\best_1024_cpu_qoperator_int8.onnx (21925621 bytes, sha256=393be321685a2cf5dbd5c92a2f2ccc92532ec97d74f3501986ed9659afafaf46)

model,threads,mean_ms,median_ms,p95_ms,fps
FP32,8,82.999,82.282,92.059,12.048
INT8,8,106.610,105.401,115.640,9.380
output_diff,threads=8,shape=(1, 300, 6),mae=285.5,max_abs=1036.16
FP32,16,76.736,76.585,78.643,13.032
INT8,16,110.115,109.075,120.006,9.081
output_diff,threads=16,shape=(1, 300, 6),mae=285.5,max_abs=1036.16
FP32,24,85.180,81.679,107.289,11.740
INT8,24,125.792,119.647,159.227,7.950
output_diff,threads=24,shape=(1, 300, 6),mae=285.5,max_abs=1036.16
```

최적값끼리 비교하면 FP32 16 threads가 76.736 ms / 13.032 FPS이고, INT8 8 threads가
106.610 ms / 9.380 FPS다. 현재 INT8 모델의 평균 지연시간이 약 38.9% 더 길다.

주의: random input의 end-to-end top-300 출력을 행 순서대로 뺀 `output_diff`는 검출 정확도 지표가 아니다.
실제 이미지의 box/class/confidence 또는 전체 validation mAP로 정확도를 판단해야 한다.

## ORT 연산자 프로파일

8 threads, 동일 random input, 10회 추론의 Node duration 합계:

```text
FP32
Conv          693.893 ms
QuickGelu      96.739 ms
Concat         48.458 ms
ReorderInput   40.920 ms
Upsample       17.267 ms
ReorderOutput  16.246 ms
Split          12.401 ms
Add            10.479 ms
Mul             6.364 ms

INT8
QLinearConv      378.356 ms
DequantizeLinear 352.115 ms
Mul              115.159 ms
QuantizeLinear    83.828 ms
Sigmoid           62.906 ms
Concat            55.474 ms
Resize            54.502 ms
Conv              49.642 ms
Transpose         17.483 ms
Split             16.165 ms
Add               13.666 ms
QuickGelu         10.563 ms
```

INT8 convolution 자체는 빨라졌다. FP32 Conv 합계는 약 69.4 ms/run이고 INT8의
QLinearConv + 남은 FP32 Conv는 약 42.8 ms/run이다. 그러나 Quantize + Dequantize만 약
43.6 ms/run이 추가되고, FP32에서 가능했던 SiLU 및 기타 연산 융합도 깨져 전체 모델은 느려진다.

## 핵심 원인

현재 export 코드가 다음과 같이 Conv만 양자화한다.

```python
op_types_to_quantize=["Conv"]
```

따라서 기본 Conv의 SiLU (`x * sigmoid(x)`)와 StarGate의 ReLU6/곱셈, Add, Concat 등이 FP32로
남아 다음 패턴이 반복된다.

```text
QLinearConv -> DequantizeLinear -> FP32 activation/Mul -> QuantizeLinear -> QLinearConv
```

그래프 조사 결과:

- QLinearConv 175개 중 151개 출력이 바로 DequantizeLinear로 연결된다.
- DequantizeLinear 출력의 직접 consumer는 Mul 122개, Sigmoid 110개, Add 15개, Clip 12개 등이다.
- 양자화 경계는 StarGate가 있는 model 6, 8, 10에도 집중되지만 StarGate 하나만의 문제는 아니다.
- Detect head의 FP32 Conv 32개는 프로파일상 약 5 ms/run이므로 주원인이 아니다.
- 주원인은 backbone/neck 전체에서 Conv만 INT8이고 activation과 elementwise 연산이 FP32인 부분 양자화다.

즉, 현재 결과만으로 `StarGate는 INT8에 부적합하다` 또는 `PTQ라서 느리다`고 결론 내리면 안 된다.
PTQ/QAT는 주로 정확도와 scale 최적화의 차이다. 같은 부분 양자화 그래프로 export하면 QAT도 속도는
거의 같을 가능성이 높다.

## 다음 우선 실험

구조 변경이나 다른 QAT 알고리즘보다 먼저 다음 모델을 만들어야 한다.

```text
입력 -> Quantize -> backbone + neck 전체 INT8 -> Dequantize -> Detect head FP32
```

Detect head `/model.31/` 제외는 유지하되 `op_types_to_quantize`를 Conv 이외의 ORT 지원 연산으로
확장한다. 로컬 ONNX Runtime 1.27.0의 `QLinearOpsRegistry`에는 다음 연산들이 포함되어 있다.

```text
Add, AveragePool, Clip, Concat, Conv, Gather, Gemm, GlobalAveragePool,
LeakyRelu, MatMul, MaxPool, Mul, Pad, Relu, Reshape, Resize, Sigmoid,
Softmax, Split, Squeeze, Transpose, Unsqueeze, Where
```

첫 후보:

```python
op_types_to_quantize=[
    "Conv",
    "Add",
    "Mul",
    "Sigmoid",
    "Clip",
    "Concat",
    "MaxPool",
    "Resize",
    "Split",
    "Reshape",
    "Transpose",
]
```

후속 절차:

1. 위 full-op PTQ 모델 생성. Detect head는 FP32 유지.
2. Q/DQ 개수와 실제 QLinear 연산자 개수를 검사.
3. Linux와 i7-13700K에서 동일 조건 latency를 측정.
4. `/home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml` 전체 validation mAP 측정.
5. 정확도가 나쁠 때 backend와 dtype을 일치시킨 실제 QAT를 수행.
6. QAT 결과도 실제 CPU integer kernel로 lowering되는지 연산자와 프로파일로 확인.
7. 위 방법으로도 느릴 때만 곱셈 gate를 제거한 `StarGateINT8` 구조를 새로 학습.

QAT 알고리즘만 LSQ, PACT 등으로 교체해도 export가 Conv-only이면 Q/DQ 비용은 해결되지 않는다.
반대로 QAT가 activation, Mul, Add와 Detect 일부까지 정확도를 유지하며 INT8화할 수 있게 해 준다면
FP32 구간을 줄이는 간접적인 속도 효과는 가능하다.

## 정확도 평가 상태

사용자 요청에 따라 Windows에서는 mAP 평가를 아직 수행하지 않았다. 지정 YAML과 그 YAML이 가리키는
dataset이 Windows 작업공간에 없고 WSL도 설치되어 있지 않다. 전체 validation mAP는 Linux 서버에서
다음 YAML로 수행한다.

```text
/home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml
```

기존 문서에 기록된 FP32/MQBench calibration/수동 LSQ 값은 현재 ORT full-op INT8 배포 모델의 mAP가
아니므로 새 모델 생성 후 반드시 다시 측정해야 한다.
