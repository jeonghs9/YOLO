# SegFormer Migration Verification — 2026-08-10

## Scope

The NVlabs SegFormer source and local lane-segmentation extensions were copied from:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/PROJECT/OCR-CAR-CATEGORY/git/SegFormer
```

to:

```text
/home/hsjeong/workspace/vision-seg/SegFormer/source
```

The legacy source, pretrained models, checkpoints, and work directories were not deleted.

## Source provenance

```text
upstream: https://github.com/NVlabs/SegFormer.git
branch at migration: master
commit: 65fa8cfa9b52b6ee7e8897a98705abf8570f9e32
```

The following local extensions were preserved:

```text
mmseg/datasets/__init__.py
mmseg/datasets/aihub_lane.py
mmseg/datasets/skyscapes.py
local_configs/segformer/B0/segformer.b0.1024x1024.skyscapes.20k.py
local_configs/segformer/B0/segformer.b0.1024x640.aihub.40k.py
local_configs/segformer/B0/segformer.b0.512x768.aihub_bev.40k.py
local_configs/segformer/B2/segformer.b2.1024x1024.skyscapes.20k.py
```

The upstream `.git` metadata was separated to the temporary recovery path:

```text
/tmp/SegFormer-source.git.backup-20260810
```

This `/tmp` path is not permanent storage. Upstream provenance is also recorded in the project README.

## Artifact policy

The legacy SegFormer directory was approximately 23GB:

```text
work_dirs:  approximately 22GB
pretrained: approximately 867MB
source and Git metadata: only a few MB
```

The large artifacts were not duplicated during the initial source verification. They were subsequently moved into the new SegFormer project, and the following ignored local symlinks provide compatibility:

```text
source/pretrained -> ../models/pretrained
source/work_dirs  -> ../outputs/work_dirs
```

The actual artifact locations are now `models/pretrained` and `outputs/work_dirs`. The old SegFormer directory no longer contains `pretrained` or `work_dirs`.

## Environment correction

Environment:

```text
/home/hsjeong/miniconda3/envs/segformer_cu111
```

Running without `PYTHONNOUSERSITE=1` loaded user-level Python 3.8 MMCV 1.7.1 and failed the MMSEG compatibility assertion. All verified commands therefore use:

```bash
PYTHONNOUSERSITE=1
```

With that setting:

```text
MMCV: 1.3.0
MMSEG: 0.11.0
MMSEG source: /home/hsjeong/workspace/vision-seg/SegFormer/source/mmseg
```

The environment's editable `mmsegmentation` install was changed to the new source path with `--no-deps`.

## Verification results

The following checks passed:

1. `mmcv.ops.RoIAlign` CUDA extension import.
2. `SkyScapesLaneDataset` and `AIHubLaneDataset` registration.
3. B0 config loading and model construction.
4. Model type `EncoderDecoder`, backbone `mit_b0`, head `SegFormerHead`, 3 classes.
5. GPU checkpoint load and inference on one real 1024×1024 SkyScapes test tile.
6. Pixel-for-pixel comparison against the legacy-path baseline.

Regression input:

```text
/home/hsjeong/workspace/dataset/etc./DLR-SkyScapes/split/tiles/test/images/2012-04-26-Muenchen-Tunnel_4K0G0080_x03072_y00000.png
```

Checkpoint:

```text
work_dirs/b0_skyscapes_20k/iter_10000.pth
```

Both legacy and new paths produced:

```text
shape: (1024, 1024)
dtype: int64
mask SHA-256: b6b82fc338b86624117490972324381da7f7b421123a752f6057a29e672dcc9d
different pixels: 0
```

## Decision

The new source path is functionally equivalent for the tested config and checkpoint. It is safe to use the new source and editable installation.

The new source and artifacts no longer depend on the legacy SegFormer paths. After the final reference and file audit, the remaining legacy source directory was deleted on 2026-08-11. The editable environment continues to import the new source path.

Artifact cleanup performed after this verification is recorded in `ARTIFACT_AUDIT_2026-08-10.md`. The generated 20.2GB `res.pkl` and approximately 144MB of test visualizations were removed, reducing the legacy SegFormer directory to approximately 3.6GB.
