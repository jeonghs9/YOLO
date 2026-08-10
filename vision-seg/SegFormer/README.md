# SegFormer Lane Segmentation

NVlabs SegFormer source with local SkyScapes and AI Hub lane-dataset extensions.

## Layout

```text
source/   upstream SegFormer source plus local datasets and configs
scripts/  dataset preparation, environment setup, conversion, and inference tools
docs/     experiment and environment records
configs/  project-level configuration index
models/   local checkpoints and pretrained weights; excluded from Git
outputs/  work directories and visualizations; excluded from Git
```

## Upstream provenance

```text
repository: https://github.com/NVlabs/SegFormer.git
commit: 65fa8cfa9b52b6ee7e8897a98705abf8570f9e32
```

The source is managed by the parent `vision-research` repository. The upstream `.git` directory was separated during migration rather than committed as a nested repository.

## Runtime

Use the dedicated environment and disable the Python user site:

```bash
PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/segformer_cu111/bin/python
```

Without `PYTHONNOUSERSITE=1`, this host may load an incompatible user-level MMCV 1.7.1 instead of the environment's MMCV 1.3.0.

Large artifacts are stored outside the Git-managed source tree:

```text
models/pretrained/   actual pretrained weights
outputs/work_dirs/   actual checkpoints and training logs
```

For compatibility with upstream configs and tools, the source tree contains local ignored relative symlinks:

```text
source/pretrained -> ../models/pretrained
source/work_dirs  -> ../outputs/work_dirs
```
