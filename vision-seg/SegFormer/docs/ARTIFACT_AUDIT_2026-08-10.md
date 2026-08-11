# SegFormer Artifact Audit — 2026-08-10

## Scope

Audited legacy artifact path:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/PROJECT/OCR-CAR-CATEGORY/git/SegFormer
```

This was the artifact path at the start of the audit. After generated-output cleanup, the retained artifacts were moved into the new `vision-seg/SegFormer` project.

## Initial usage

```text
SegFormer total: approximately 23GB
work_dirs:       approximately 22GB
pretrained:      approximately 867MB
```

No MP4, AVI, MOV, MKV, or WebM files existed under `work_dirs`.

The apparent 22GB experiment usage was primarily one generated MMSEG output cache:

```text
work_dirs/res.pkl: 20,205,505,122 bytes
```

`tools/test.py` uses `work_dirs/res.pkl` as its default `--out` result path. No project source, config, or documentation depended on this generated file.

## Removed generated outputs

The user authorized removal of generated video and similar output artifacts. The following reproducible outputs were deleted:

```text
work_dirs/res.pkl
work_dirs/b0_skyscapes_20k/vis_test/
work_dirs/b2_skyscapes_30k/vis_test/
```

The two `vis_test` directories contained 96 PNG and 4 JPG visualization files and occupied approximately 144MB combined.

After removal:

```text
work_dirs:       approximately 2.7GB
SegFormer total: approximately 3.6GB
```

The deleted data is not recoverable from the filesystem but can be regenerated from the retained checkpoints, configs, source, and datasets.

## Checkpoint selection

Best checkpoints were selected by maximum validation `mIoU` recorded in each JSON log. Final checkpoints are retained separately when different from the best checkpoint.

| Experiment | Best checkpoint | Best mIoU | Final checkpoint | Keep |
|---|---:|---:|---:|---|
| B0 SkyScapes | `iter_7000.pth` | 0.5568 | `iter_10000.pth` | best + final |
| B2 SkyScapes | `iter_14000.pth` | 0.5838 | `iter_30000.pth` | best + final |
| B0 AI Hub | `iter_40000.pth` | 0.6513 | `iter_40000.pth` | one file |
| B0 AI Hub BEV | `iter_38000.pth` | 0.6609 | `iter_40000.pth` | best + final |

Recommended checkpoint retention set:

```text
b0_skyscapes_20k/iter_7000.pth
b0_skyscapes_20k/iter_10000.pth
b2_skyscapes_30k/iter_14000.pth
b2_skyscapes_30k/iter_30000.pth
b0_aihub_40k/iter_40000.pth
b0_aihub_bev_40k/iter_38000.pth
b0_aihub_bev_40k/iter_40000.pth
```

The seven retained checkpoint files occupy approximately 0.82GiB.

The other 38 real intermediate checkpoint files are deletion candidates and occupy approximately 1.85GiB:

```text
B0 AI Hub:     every 2,000-step checkpoint except iter_40000
B0 AI Hub BEV: every 2,000-step checkpoint except iter_38000 and iter_40000
B2 SkyScapes:  iter_17000.pth
```

Each `latest.pth` entry is a symbolic link to its final checkpoint and does not duplicate checkpoint storage.

## Pretrained model selection

Current local configs use B0 and B2 backbones. Recommended retention:

```text
mit_b0.pth    12.7MiB
mit_b2.pth    92.4MiB
```

The following unused pretrained files are deletion candidates, totaling approximately 0.74GiB:

```text
mit_b1.pth    50.2MiB
mit_b3.pth   168.3MiB
mit_b4.pth   232.3MiB
mit_b5.pth   311.0MiB
```

They are downloadable or reproducible, but were not deleted during this audit.

## Potential final footprint

If the intermediate checkpoint and unused pretrained candidates are removed, the preserved SegFormer artifacts should be close to 1GB plus small logs and configs.

No checkpoint or pretrained model deletion should occur without explicit confirmation.

## Current checkpoint state — 2026-08-11

A follow-up audit found that intermediate checkpoints had already been removed. Exactly one real checkpoint remains for each experiment:

| Experiment | Retained checkpoint | Type according to validation log | Recorded mIoU |
|---|---:|---|---:|
| B0 AI Hub | `iter_40000.pth` | best and final | 0.6513 |
| B0 AI Hub BEV | `iter_40000.pth` | final; logged best was iter 38000 | 0.6608 |
| B0 SkyScapes | `iter_10000.pth` | final; logged best was iter 7000 | 0.5534 |
| B2 SkyScapes | `iter_14000.pth` | best | 0.5838 |

The four checkpoints, logs, and copied configs occupy approximately 443MB. There are no remaining intermediate checkpoint deletion candidates.

The pretrained directory still contains `mit_b0.pth` through `mit_b5.pth` and occupies approximately 867MB. Per user request, no pretrained file was removed.

## Artifact relocation

After the audit, the remaining artifact directories were moved without duplication:

```text
legacy SegFormer/pretrained
  -> /home/hsjeong/workspace/vision-seg/SegFormer/models/pretrained

legacy SegFormer/work_dirs
  -> /home/hsjeong/workspace/vision-seg/SegFormer/outputs/work_dirs
```

Source compatibility links now use relative paths:

```text
source/pretrained -> ../models/pretrained
source/work_dirs  -> ../outputs/work_dirs
```

GPU inference from the relocated checkpoint produced the same mask SHA-256 as the original baseline, with zero differing pixels.

After reference verification, the remaining legacy SegFormer source directory was deleted on 2026-08-11. Legacy UTIL scripts that still referenced the old source path were updated to `/home/hsjeong/workspace/vision-seg/SegFormer/source` before deletion.
