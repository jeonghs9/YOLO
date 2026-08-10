# Legacy UTIL File Disposition

Source reviewed:

```text
/home/hsjeong/workspace/Yolo26/ultralytics/PROJECT/OCR-CAR-CATEGORY/UTIL
```

No legacy files were deleted during this review.

## Current deployment release

Keep and use:

```text
visualize_solid_lane_crossing_v5.3.1.py
```

It is standalone with respect to project-local Python imports. Its required runtime dependencies are external Python packages, two model files, an input video, and the Ultralytics tracker configuration.

## Move with SegFormer research

These files are not required by the current deployment release, but are required to reproduce or continue SegFormer work. Move them with the future `vision-seg/SegFormer` project rather than deleting them.

```text
aihub_make_bev.py
aihub_make_masks.py
aihub_make_split.py
aihub_visualize.py
mit_hf_to_nvlabs.py
restore_user_site_py38.sh
setup_segformer_env.sh
skyscapes_make_split.py
skyscapes_make_tiles.py
skyscapes_video_inference.py
skyscapes_visualize_test.py
visualize_skyscapes_lane_labels.py
```

Also preserve the corresponding SegFormer documents and local configs before changing the SegFormer repository path.

## Keep as application development tools

These are not imported by v5.3.1, but remain useful for dataset preparation, model validation, OCR diagnosis, or sample collection. They should be reviewed and moved into `scripts/` or `tests/` rather than immediately deleted.

```text
dataset_prepare_vehicle_cls_split.py
test_lane_seg_directional_roi.py
test_lane_seg_instance_count.py
track_save_plate_crops.py
visualize_car_lane_labels.py
yolo_lane_seg_video.py
```

## Historical OCR and motion prototypes

These files are superseded for the final solid-lane deployment path, but contain prior OCR, association, optical-flow, and vehicle-classification experiments. They are candidates for a compressed archive or a dedicated historical Git commit before deletion.

```text
ocr_plate_fast_plate.py
ocr_plate_optical_flow_lane_change_v3.py
ocr_plate_paddle_all_classes_vehicle_cls.py
ocr_plate_paddle_basic.py
ocr_plate_paddle_car_assoc.py
ocr_plate_paddle_car_assoc_formatted_final.py
ocr_plate_paddle_car_assoc_vehicle_cls.py
visualize_optical_flow_debug_v2.2.1.py
visualize_optical_flow_debug_v2.3.py
```

## Historical version archive

The following existing archive directories are not runtime dependencies of v5.3.1:

```text
archive/lane_crossing/
archive/ocr_optical_flow/
```

They are deletion candidates only after their history is stored in Git or a recoverable archive. They should not be mixed into the new deployment source tree.

## Documentation to preserve

At minimum, preserve:

```text
md/LANE_POSTPROCESS_INDEX.md
md/LANE_POSTPROCESS_05_3_UNKNOWN_GAP_RECOVERY.md
```

The remaining `md/` files document experiments and design decisions. They can be moved into topic-specific documentation or archived, but should not be treated as runtime clutter and deleted solely because v5.3.1 does not import them.

## Deletion gate

Do not delete the legacy `UTIL` directory until all of the following are true:

1. SegFormer files and configs are preserved under `vision-seg`.
2. Selected data-preparation and validation utilities are relocated.
3. Historical prototypes are committed or archived.
4. The full representative-video regression passes from the new location.
5. OCR cache and model paths no longer depend on an old directory that will be removed.

