# Windows Codex 작업 인수인계

## 새 Codex 세션에서 가장 먼저 할 일

다음 요청으로 시작한다.

```text
YOLO26_QUANTIZATION_01_OVERVIEW.md부터 YOLO26_QUANTIZATION_04_WINDOWS_RESULTS.md까지 읽고
현재 상태를 설명한 뒤,
Windows의 i7-13700K에서 FP32와 CPU INT8 ONNX 모델의 속도를 동일 조건으로 비교해줘.
기존 파일을 변경하기 전 git status를 확인하고 사용자 변경사항은 건드리지 마.
```

## 저장소

- GitHub: `https://github.com/jeonghs9/ultralytics-yolo26.git`
- 브랜치: `STAR_GATE-MQBench`
- CPU INT8 변환 코드 최초 커밋: `2c85347`
- 서버 저장소: `/home/hsjeong/workspace/Yolo26/ultralytics/ultralytics`
- 원본 모델: `/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best.pt`
- 데이터 YAML: `/home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml`

## 목적

정확도가 충분한 YOLO26 StarGate 객체 검출 모델의 CPU 추론 속도를 높인다. 양자화 알고리즘의
정확도 보존 효과와 실제 CPU INT8 커널의 속도 효과를 구분해서 평가한다.

## MQBench 적용 상태

- 서버 환경: `/home/hsjeong/miniconda3/envs/mqbench_env`
- MQBench 소스: `/home/hsjeong/tmp/MQBench`
- `prepare_by_platform` 성공
- YOLO26 FX 추적을 위해 wrapper, `forward_split`, activation deepcopy, leaf module 처리를 적용
- FakeQuantize 453개 삽입 확인
- W8A8 구성: weight per-channel symmetric, activation per-tensor
- 256장 calibration 후 fake-quant 정확도 확인

정확도 결과:

```text
FP32                    mAP50-95 0.5057, mAP50 0.7643
MQBench calibration PTQ mAP50-95 0.5029, mAP50 0.7660
수동 LSQ QAT            mAP50-95 0.4993, mAP50 0.7596
```

주의: 기존 실험은 `LearnableFakeQuantize`를 사용했지만 scale 최적화 학습은 하지 않았다. 따라서
엄밀한 LSQ 학습 결과라기보다 LSQ fake quantizer를 사용한 calibration PTQ 결과다.

## MQBench QDQ 실패 원인

MQBench QDQ ONNX를 ONNX Runtime CPU로 실행했을 때 ORT가 Conv를 실제 정수 Conv로 융합하지
못했다.

```text
FP32 ONNX       159.7 ms, 6.26 FPS
MQBench QDQ     332.4 ms, 3.01 FPS
QLinearConv     0개
```

FP Conv가 남고 Quantize/Dequantize 오버헤드만 추가된 것이 원인이다.

## 실제 CPU INT8 변환

재현 코드:

- `utils/quantization/mqbench_adapter.py`
- `utils/quantization/export_ort_cpu_int8.py`
- `/home/hsjeong/workspace/md/quantization/YOLO26_QUANTIZATION_02_MQBENCH_CPU_INT8.md`

생성된 서버 모델:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/best_1024.onnx
/home/hsjeong/workspace/Yolo26/ultralytics/models/3차모델/ONNX/best_1024_cpu_qoperator_int8.onnx
```

CPU INT8 모델 상태:

```text
300장 calibration
QLinearConv 175개
Detect head FP32 Conv 32개
파일 크기 약 21 MB
CPUExecutionProvider smoke inference 통과
실제 이미지에서 FP32 5 boxes, INT8 4 boxes
```

이 QOperator 모델은 실제 CPU INT8 연산을 수행하지만 MQBench LSQ scale을 그대로 보존하지 않는다.
MQBench 정확도 실험과 ORT CPU 배포 변환은 현재 별도 단계다.

## 서버 CPU 결과

서버 CPU는 Intel Xeon Gold 6148 2소켓, 총 40 cores/80 threads다. AVX-512는 지원하지만 VNNI는
지원하지 않는다.

20 ORT threads, batch 1, 1024 입력의 짧은 네트워크 벤치:

```text
FP32  190.88 ms, 5.24 FPS
INT8  210.45 ms, 4.75 FPS
```

INT8 변환은 성공했지만 이 서버에서는 약 9% 느렸다. INT8 모델 오류가 아니라 구형 CPU의 정수
커널 효율, INT8/FP32 변환, FP32 Detect head, 연산 융합 제한의 영향으로 해석한다.

## Windows 로컬 테스트 목표

로컬 CPU는 Intel Core i7-13700K다. 다음 조건을 반드시 동일하게 맞춘다.

- 동일 입력 tensor를 FP32와 INT8 세션에 사용
- `CPUExecutionProvider`만 사용
- batch 1, 1024x1024
- warmup 후 최소 100회 측정
- 세션 생성 시간과 이미지 전처리는 네트워크 지연시간에서 제외
- `intra_op_num_threads`를 8, 16, 24로 각각 비교
- 평균, median, p95, FPS 기록
- Windows 전원 모드를 고성능으로 설정
- 모델 정확성 확인을 위해 동일 이미지의 box 수, class, confidence도 비교

Windows 모델 권장 경로:

```text
C:\workspace\models\best_1024.onnx
C:\workspace\models\best_1024_cpu_qoperator_int8.onnx
```

## 다음 양자화 후보

현재 MQBench 소스의 advanced PTQ는 다음을 지원한다.

1. AdaRound
2. BRECQ
3. QDrop

권장 실험 순서는 AdaRound baseline, BRECQ, QDrop이다. 이 알고리즘들은 주로 양자화 정확도를
개선한다. 모두 같은 W8A8 연산 그래프로 배포되면 CPU 속도는 거의 같으며, 실제 속도는 런타임과
CPU 정수 명령어 지원에 의해 결정된다.

## 주의사항

- 서버의 모델 파일과 데이터셋은 Git에 포함되지 않는다. Windows로 별도 복사해야 한다.
- Git의 기존 미추적 파일을 삭제하거나 되돌리지 않는다.
- Windows 결과와 Linux 서버의 절대 FPS를 직접 비교하지 않는다. Windows 내부에서 FP32/INT8
  비율을 비교한다.
- 전체 데이터셋 mAP 검증 전에는 한 이미지 결과만으로 배포 가능하다고 판단하지 않는다.
