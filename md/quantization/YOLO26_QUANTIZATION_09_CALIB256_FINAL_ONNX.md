# YOLO26 Advanced PTQ: Calibration 256 Final ONNX

## Scope

- Source model: `/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt`
- Dataset: `/home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml`
- Input size: 1024
- Quantization: W8A8
- PTQ calibration: 256 train images, hybrid selection, seed 0
- Hybrid selection: 50% seeded random and 50% class/object-size coverage
- Reconstruction: 20,000 updates per block or layer
- ORT runtime calibration: 512 train images, hybrid selection, seed 0
- Test-set validation: not run; the user will validate the final files

The 256-image reconstruction setting follows the object-detection calibration size used by BRECQ and QDrop experiments. The earlier 64-image outputs remain comparison artifacts.

## Memory-safe reconstruction

At 1024 resolution, keeping 256 inputs and all captured intermediate features on a 16 GB V100 is unsafe. The AdaRound, BRECQ, and QDrop runners were changed to:

1. Keep calibration images and captured features in CPU memory.
2. Transfer only the randomly selected reconstruction sample to the GPU.
3. Preserve all 256 samples as reconstruction candidates.
4. Keep the original 20,000 optimization updates.

This changes storage and transfer behavior only; it does not reduce the calibration set or optimization count.

## Algorithm runs

### BRECQ

- GPU: physical GPU 2
- Reconstruction: 22 YOLO backbone/neck blocks
- Updates: 20,000 per block
- Warm-up: 0.2
- Rounding loss weight: 0.01
- MQBench checkpoint: `best_1024_brecq_w8a8_calib256_iter20000_mqbench.pt`

### QDrop

- GPU: physical GPU 3
- Reconstruction: 22 YOLO backbone/neck blocks
- Updates: 20,000 per block
- Drop probability: 0.5
- Activation scale learning rate: `4e-5`
- Warm-up: 0.2
- Rounding loss weight: 0.01
- MQBench checkpoint: `best_1024_qdrop_w8a8_calib256_iter20000_mqbench.pt`

### AdaRound

- GPU: physical GPU 4
- Reconstruction: 175 Conv layers
- Updates: 20,000 per layer
- Detect-head quantizers excluded: 64
- Warm-up: 0.2
- Rounding loss weight: 0.01
- MQBench checkpoint metadata verified: calibration 256, updates 20,000, layers 175
- MQBench checkpoint: `best_1024_adaround_w8a8_calib256_iter20000_mqbench.pt`

## Final CPU INT8 ONNX files

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/AdaRound/best_1024_adaround_w8a8_calib256_iter20000_fullops.onnx
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/BRECQ/best_1024_brecq_w8a8_calib256_iter20000_fullops.onnx
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/QDrop/best_1024_qdrop_w8a8_calib256_iter20000_fullops.onnx
```

Each file is approximately 21 MB. The ORT conversion reported the following graph structure for every file:

- QLinearConv: 175
- FP32 Conv retained: 32
- QLinearAdd: 40
- QLinearConcat: 28
- QLinearMul: 123
- QLinearSigmoid: 110
- QuantizeLinear: 4
- DequantizeLinear: 8

All three final files successfully created an ONNX Runtime CPU session and completed one smoke inference with one output tensor.
