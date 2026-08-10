# TensorRT Engine Export Notes

## Context

- Target project: `/home/hsjeong/workspace/Yolo26/ultralytics`
- Weight file: `/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt`
- Successful 640 engine file: `/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best_640.engine`
- Successful 1024 engine file: `/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best_1024.engine`
- `best.engine` is the most recently exported engine and may be overwritten by the next export.

## Environment

- Conda env: `/home/hsjeong/miniconda3/envs/Yolov26_env_3.10`
- Python: `3.10`
- Ultralytics install mode: editable install from `/home/hsjeong/workspace/Yolo26/ultralytics`
- PyTorch: `2.9.1+cu128`
- TensorRT: `8.6.1`

## Why This Setup Is Needed

- The server GPUs are Tesla V100, compute capability SM70.
- TensorRT `10.13.3.9.post1` failed because that release does not support SM70.
- TensorRT `8.6.1` supports V100, but it needs `libcudnn.so.8`.
- Torch `2.9.1` needs cuDNN 9, so cuDNN 8 is kept separately at:

```bash
/home/hsjeong/miniconda3/envs/Yolov26_env_3.10/tensorrt_cudnn8/lib
```

## Successful 640 Export Command

```bash
cd /home/hsjeong/workspace/Yolo26/ultralytics

LD_LIBRARY_PATH=/home/hsjeong/miniconda3/envs/Yolov26_env_3.10/tensorrt_cudnn8/lib:/home/hsjeong/miniconda3/envs/Yolov26_env_3.10/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/hsjeong/miniconda3/envs/Yolov26_env_3.10/lib/python3.10/site-packages/tensorrt_libs:$LD_LIBRARY_PATH \
/home/hsjeong/miniconda3/envs/Yolov26_env_3.10/bin/python -c \
'from ultralytics import YOLO; YOLO("/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt").export(format="engine", imgsz=640, half=True, device=4, opset=17)'
```

## Successful 1024 Export Command

```bash
cd /home/hsjeong/workspace/Yolo26/ultralytics

LD_LIBRARY_PATH=/home/hsjeong/miniconda3/envs/Yolov26_env_3.10/tensorrt_cudnn8/lib:/home/hsjeong/miniconda3/envs/Yolov26_env_3.10/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/hsjeong/miniconda3/envs/Yolov26_env_3.10/lib/python3.10/site-packages/tensorrt_libs:$LD_LIBRARY_PATH \
/home/hsjeong/miniconda3/envs/Yolov26_env_3.10/bin/python -c \
'from ultralytics import YOLO; YOLO("/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt").export(format="engine", imgsz=1024, half=True, device=5, opset=17)'
```

## Outputs

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best_640.engine
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best_1024.engine
```

Both engines were built as FP16 TensorRT engines. The plain `best.engine` file is overwritten on each export, so copy or rename it after each build.
