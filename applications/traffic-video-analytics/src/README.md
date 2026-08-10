# Source layout

Future behavior-preserving extraction targets:

- `detection/`: vehicle and plate detection adapters
- `segmentation/`: lane segmentation and mask processing
- `ocr/`: plate OCR and text stabilization
- `tracking/`: vehicle tracking state
- `event_detection/`: solid-lane crossing decisions
- `pipeline/`: end-to-end orchestration

The verified v5.3.1 release remains in `../scripts/` until regression tests cover the full pipeline.

