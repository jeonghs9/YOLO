# MQBench and CPU INT8 deployment

## Scope

MQBench LSQ is used to prepare and evaluate the quantization behavior of the
YOLO26 model. Its QDQ export is not directly executable as integer convolution
by ONNX Runtime CPU for this graph: ORT retains floating-point `Conv` nodes and
adds quantize/dequantize overhead.

The CPU deployment step therefore uses ONNX Runtime static calibration and the
QOperator format. A successful artifact must contain `QLinearConv` operators.
The Detect head remains FP32 because quantizing its output path previously
produced invalid all-zero detection metrics.

This distinction is important: the QOperator deployment artifact is true CPU
INT8, but it does not preserve MQBench LSQ activation scales exactly. MQBench
accuracy experiments and ORT CPU deployment are separate stages.

## Export

Run from the repository root with the `mqbench_env` environment:

```bash
python -m utils.quantization.export_ort_cpu_int8 \
  --model-input /home/hsjeong/tmp/mqbench_onnx/stargate_fp32.onnx \
  --model-output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/ONNX/best_1024_cpu_qoperator_int8.onnx \
  --data-yaml /home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml \
  --split train \
  --imgsz 1024 \
  --max-calib-images 300
```

The command fails if no `QLinearConv` is created or if CPU smoke inference
returns invalid values. Accuracy and latency must still be measured on the full
validation set before deployment.

## Verified result (2026-06-30)

- Calibration images: 300 at 1024x1024
- Integer operators: 175 `QLinearConv`
- Detect head retained in FP32: 32 `Conv`
- Output size: 21 MB
- CPU smoke inference: passed
- One-image YOLO check: FP32 5 boxes, INT8 4 boxes
- Network-only latency, 20 ORT threads: FP32 190.88 ms, INT8 210.45 ms

The INT8 graph is executable, but it is not faster on the current Xeon Gold
6148. This CPU has AVX-512 but no VNNI, and the FP32 Detect head plus quantize
boundaries add overhead. Do not treat INT8 operator conversion alone as proof of
a latency improvement; validate on the target customer CPU.

## Full-op backbone and neck experiment (2026-06-30)

Conv-only quantization left activation and elementwise operators in FP32. The
resulting 130 QuantizeLinear and 151 DequantizeLinear nodes dominated runtime.
The exporter now accepts `--op-types`, allowing the feature path to remain INT8
while `/model.31/` Detect stays FP32:

```bash
python -m utils.quantization.export_ort_cpu_int8 \
  --model-input /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best_1024.onnx \
  --model-output /home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/ONNX/best_1024_cpu_qoperator_int8_fullops.onnx \
  --data-yaml /home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml \
  --split train --imgsz 1024 --max-calib-images 300 \
  --op-types Conv Add Mul Sigmoid Clip Concat MaxPool Resize Split Reshape Transpose
```

Generated graph:

```text
QLinearConv 175, QLinearAdd 40, QLinearMul 123
QLinearSigmoid 110, QLinearConcat 28
QuantizeLinear 10, DequantizeLinear 11, FP32 Detect Conv 32
```

Xeon Gold 6148, ORT 1.23.2, 20 threads, 50 measured runs:

```text
FP32          167.736 ms, 5.962 FPS
Conv-only INT8 249.083 ms, 4.015 FPS
Full-op INT8   103.142 ms, 9.695 FPS
```

Full test split validation (6,701 images):

```text
Recorded FP32 baseline: mAP50-95 0.5057, mAP50 0.7643
Full-op INT8:           mAP50-95 0.4390, mAP50 0.7076
```

Full-op INT8 is 1.63x faster than FP32 on this server, but its 0.0667 absolute
mAP50-95 loss is too large for deployment without further accuracy recovery.
The next experiment should target activation and gate quantization accuracy
using improved calibration or reconstruction/QAT, while preserving this
low-boundary deployment graph.
