# AI-TOD v1 데이터셋 준비 기록

## 요약

- 데이터셋 버전: AI-TOD v1
- 원본 루트: `/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1`
- YOLO 학습용 루트: `/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/cleaning`
- YOLO YAML: `/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/cleaning/ai-tod.yaml`
- xView 원본 이미지 용량: 약 23GB
- 생성된 AI-TOD 이미지 용량: 약 38GB
- YOLO 정리본 용량: 약 38GB
- 전체 v1 디렉터리 용량: 약 125GB

## 원본 경로

```text
/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/
  xview/ori/train_images/
  xview/ori/xView_train.geojson
  AI-TOD_wo_xview/images_wo_xview/
  AI-TOD_wo_xview/complete_annotations/
  aitod/images/
```

세부 원본 파일은 다음과 같다.

| 항목 | 경로 |
| --- | --- |
| xView 원본 이미지 | `/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/xview/ori/train_images/` |
| xView annotation | `/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/xview/ori/xView_train.geojson` |
| AI-TOD without xView 이미지 zip | `/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/AI-TOD_wo_xview/images_wo_xview/` |
| AI-TOD COCO annotation | `/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/AI-TOD_wo_xview/complete_annotations/` |
| 생성/통합 이미지 | `/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/aitod/images/` |

이미지 zip 파일:

```text
aitod_wo_xview_train_imgs.zip
aitod_wo_xview_val_img.zip
aitod_wo_xview_test_imgs.zip
aitod_wo_xview_trainval_imgs.zip
```

COCO annotation 파일:

```text
aitod_train.json
aitod_val.json
aitod_test_v1_1.0.json
aitod_trainval_v1_1.0.json
```

## 전처리 절차

### 1. xView 이미지 다운로드

사용자가 xView 원본 tif 이미지를 아래 경로에 다운로드했다.

```text
/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/xview/ori/train_images/
```

### 2. AI-TOD toolkit 실행

AI-TOD toolkit 위치:

```text
/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/AI-TOD/aitodtoolkit
```

실행 명령:

```bash
cd /home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/AI-TOD/aitodtoolkit
python generate_aitod_imgs.py
```

실행 중 수정한 내용:

- `mmcv` import 실패: 실제 사용하지 않는 import라 제거했다.
- `wwtool` 패키지 없음: annotation 로딩에 필요한 최소 local module을 추가했다.
- xView annotation 경로 불일치: `xview/ori/xView_train.geojson`도 찾도록 수정했다.
- 일부 중간 tile 이미지 없음: xView 원본 tif에서 직접 crop해서 tile을 생성하는 fallback을 추가했다.
- toolkit 내부 경로 연결:
  - `aitodtoolkit/xview -> ../../v1/xview`
  - `aitodtoolkit/aitod/images -> ../../../v1/aitod/images`

### 3. AI-TOD without xView 이미지 압축 해제

AI-TOD without xView zip 파일을 split별 이미지 디렉터리에 풀었다.

```bash
unzip aitod_wo_xview_train_imgs.zip -d /home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/aitod/images/train
unzip aitod_wo_xview_val_img.zip -d /home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/aitod/images/val
unzip aitod_wo_xview_test_imgs.zip -d /home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/aitod/images/test
unzip aitod_wo_xview_trainval_imgs.zip -d /home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/v1/aitod/images/trainval
```

### 4. COCO annotation을 YOLO label로 변환

변환 스크립트:

```text
/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/cleaning/convert_coco_to_yolo.py
```

실행 명령:

```bash
python /home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/cleaning/convert_coco_to_yolo.py
```

변환 방식:

- COCO bbox `[x, y, w, h]`를 YOLO bbox `[cx, cy, w, h]` normalized 형식으로 변환했다.
- COCO category id `1..8`을 YOLO class id `0..7`로 변환했다.
- bbox가 이미지 경계를 벗어나는 경우 image boundary 안으로 clip했다.
- invalid bbox는 skip하도록 구현했지만, 최종 결과에서 skip된 bbox는 0개였다.
- 이미지가 없는 항목을 검사했으며, 최종 결과에서 missing image는 0개였다.
- empty label file도 만들 수 있게 구현했지만, 최종 결과에서 empty label은 0개였다.

초기에는 `cleaning/{split}/images`를 symlink로 만들었다. 그러나 Ultralytics가 symlink 실제 경로 기준으로 label path를 계산하면서 학습 로그의 `instances`가 0으로 잡히는 문제가 있었다. 그래서 최종 구조에서는 이미지를 `cleaning/{split}/images`에 실제 파일로 복사했다.

## 최종 YOLO 구조

```text
/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/cleaning/
  ai-tod.yaml
  convert_coco_to_yolo.py
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
  test/labels/
  trainval/images/
  trainval/labels/
```

YAML 내용:

```yaml
path: /home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/cleaning
train: train/images
val: val/images
test: test/images

names:
  0: airplane
  1: bridge
  2: storage-tank
  3: ship
  4: swimming-pool
  5: vehicle
  6: person
  7: wind-mill
```

학습 시 사용할 경로:

```bash
data=/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/cleaning/ai-tod.yaml
```

예시:

```bash
yolo detect train \
  model=yolo26n.yaml \
  data=/home/hsjeong/workspace/SOD-PAPER/dataset/AI-TOD/cleaning/ai-tod.yaml \
  imgsz=640
```

`trainval` 전체를 학습에 쓰려면 별도 YAML을 만들거나 `ai-tod.yaml`의 `train` 값을 다음처럼 바꾼다.

```yaml
train: trainval/images
```

## 클래스 매핑

| YOLO id | class name |
| ---: | --- |
| 0 | airplane |
| 1 | bridge |
| 2 | storage-tank |
| 3 | ship |
| 4 | swimming-pool |
| 5 | vehicle |
| 6 | person |
| 7 | wind-mill |

## 최종 데이터 수

| split | source images | YOLO images | YOLO label files | boxes | empty labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 11214 | 11214 | 11214 | 282580 | 0 |
| val | 2804 | 2804 | 2804 | 70424 | 0 |
| test | 14018 | 14018 | 14018 | 347617 | 0 |
| trainval | 14018 | 14018 | 14018 | 353004 | 0 |

참고:

- `trainval`은 `train + val` split이다.
- `test`도 annotation JSON이 있어 YOLO label을 생성했다.
- 기본 학습 YAML은 `train`과 `val`을 사용한다.

## 검증 결과

다음 검증을 완료했다.

- annotation JSON의 image 수와 실제 이미지 파일 수 일치
- `v1/aitod/images/{split}` 이미지 수와 `cleaning/{split}/images` 이미지 수 일치
- split별 YOLO label 파일 수와 image 수 일치
- YOLO label line format 검사 통과
- class id 범위 `0..7` 검사 통과
- normalized bbox 값 범위 `0..1` 검사 통과
- missing image 0개
- skipped bbox 0개
- empty label 0개
- `convert_coco_to_yolo.py` Python compile 검사 통과

## 주의 사항

- `cleaning/{split}/images`는 symlink가 아니라 실제 이미지 복사본이어야 한다.
- symlink로 둘 경우 Ultralytics가 label path를 원본 이미지 경로 기준으로 계산해서 `instances=0`이 발생할 수 있다.
- 기존에 symlink 상태에서 만든 run은 정상 학습 결과로 쓰면 안 된다.
