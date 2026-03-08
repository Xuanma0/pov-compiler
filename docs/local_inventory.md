# Local Inventory

## Scope

- Probed local paths:
  - `D:\BYES`
  - `D:\Ego4D_Dataset`
  - `D:\ILSVRC2017_DET_test_new`
- No runtime code or scripts were modified.

## Repo Inventory

| Repo Root | Git Repo | README | requirements | pyproject | Notes |
|---|---:|---:|---:|---:|---|
| `D:\BYES\repos\Depth-Anything-3` | yes | yes | yes | yes | Any-view depth / pose / DA3-Streaming; strongest geometry candidate |
| `D:\BYES\repos\sam3` | yes | yes | no root `requirements.txt` found | yes | Open-vocabulary segmentation + tracking; explicit Ego4D eval config present |

### Directly useful repo entry points

- `D:\BYES\repos\Depth-Anything-3\src\depth_anything_3\api.py`
- `D:\BYES\repos\Depth-Anything-3\src\depth_anything_3\cli.py`
- `D:\BYES\repos\Depth-Anything-3\da3_streaming\da3_streaming.py`
- `D:\BYES\repos\sam3\sam3\model_builder.py`
- `D:\BYES\repos\sam3\sam3\model\sam3_video_predictor.py`
- `D:\BYES\repos\sam3\sam3\train\configs\silver_image_evals\sam3_silver_image_ego4d.yaml`

## Model / Weight Inventory

| Resource | Path | Size | Category | Notes |
|---|---|---:|---|---|
| YOLO26n | `D:\BYES\models\yolo26\yolo26n.pt` | 5.29 MB | detection | Smallest detector candidate; easiest first integration |
| YOLO26s | `D:\BYES\models\yolo26\yolo26s.pt` | 19.48 MB | detection | Larger detector candidate |
| SAM3 checkpoint | `D:\BYES\models\sam3\sam3\sam3.pt` | 3290.24 MB | segmentation / tracking | Pairs with local `sam3` repo |
| SAM3 safetensors | `D:\BYES\models\sam3\sam3\model.safetensors` | 3280.58 MB | segmentation / tracking | Same family; format choice depends on load path |
| DA3 Large | `D:\BYES\models\da3\DA3-LARGE-1.1\model.safetensors` | 1567.69 MB | depth / pose | Relative depth + pose estimation |
| DA3 Nested Giant-Large | `D:\BYES\models\da3\DA3NESTED-GIANT-LARGE-1.1\model.safetensors` | 6446.42 MB | depth / geometry | Highest-capacity DA3 geometry model |
| PaddleOCR-VL-1.5 | `D:\BYES\models\OCR\PaddleOCR-VL-1.5\model.safetensors` | 1828.44 MB | OCR / document VLM | Lower priority for current POV path |
| Whisper Large v3 | `D:\BYES\models\whisper\Whisper-large-v3\large-v3.pt` | 2944.35 MB | ASR | Useful only if audio/transcript enters scope later |

### Keyword-confirmed candidates

- `YOLO26n`: `D:\BYES\models\yolo26\yolo26n.pt`
- `YOLO26s`: `D:\BYES\models\yolo26\yolo26s.pt`
- `SAM3` repo: `D:\BYES\repos\sam3`
- `SAM3` checkpoint: `D:\BYES\models\sam3\sam3\sam3.pt`
- `Depth Anything 3` repo: `D:\BYES\repos\Depth-Anything-3`
- `Depth Anything 3` weights:
  - `D:\BYES\models\da3\DA3-LARGE-1.1\model.safetensors`
  - `D:\BYES\models\da3\DA3NESTED-GIANT-LARGE-1.1\model.safetensors`

## Dataset Structure Summary

### Ego4D

Root: `D:\Ego4D_Dataset`

- `ego4d.json`
  - size: 85.05 MB
- `Ego4D_info.pth`
  - size: 1.41 MB
- `v2_packed\annotations`
  - 34 annotation files
- `v2_packed\full_scale_0000\full_scale`
  - 79 `.mp4` files
  - total size: about 48.04 GB

Representative annotation files:

- `D:\Ego4D_Dataset\v2_packed\annotations\all_narrations_redacted.json`
- `D:\Ego4D_Dataset\v2_packed\annotations\fho_main.json`
- `D:\Ego4D_Dataset\v2_packed\annotations\fho_sta_test_unannotated.json`
- `D:\Ego4D_Dataset\v2_packed\annotations\av_train.json`
- `D:\Ego4D_Dataset\v2_packed\annotations\av_val.json`

### ILSVRC2017_DET_test_new

Root: `D:\ILSVRC2017_DET_test_new`

- `ILSVRC\Data\DET\test`
  - 5500 JPEG images
- `ILSVRC\ImageSets\DET\test.txt`
  - present
  - size: 1,939,394 bytes

Representative reusable paths:

- `D:\ILSVRC2017_DET_test_new\ILSVRC\Data\DET\test`
- `D:\ILSVRC2017_DET_test_new\ILSVRC\ImageSets\DET\test.txt`

## Resources Most Likely Reusable by This Project

### Tier 1

1. `D:\BYES\models\yolo26\yolo26n.pt`
   - Best first detector candidate for a real perception backend due to small size and likely low integration friction.
2. `D:\BYES\repos\sam3`
   - Strongest segmentation / tracking repo; already contains video predictor and Ego4D-related eval config.
3. `D:\BYES\models\sam3\sam3\sam3.pt`
   - Natural checkpoint pair for the SAM3 repo.

### Tier 2

4. `D:\BYES\repos\Depth-Anything-3\da3_streaming\da3_streaming.py`
   - Directly relevant for long-video geometry under memory budget.
5. `D:\BYES\models\da3\DA3-LARGE-1.1\model.safetensors`
   - Practical DA3 geometry checkpoint.
6. `D:\Ego4D_Dataset\v2_packed\full_scale_0000\full_scale`
   - Real video source pool for future perception / retrieval / pilot validation.

## Recommended Priority Order

1. `YOLO26n / YOLO26s`
   - Fastest path to real object detections for perception/object-memory experiments.
2. `SAM3`
   - Best next step if open-vocabulary segmentation or tracking is the main target.
3. `DA3`
   - Valuable if v1.52 needs depth/geometry or long-video streaming geometry.
4. `Ego4D`
   - Primary real video source once perception backend is usable.
5. `ILSVRC2017_DET_test_new`
   - Good detector-only smoke/benchmark set, but weaker fit than Ego4D for end-to-end POV behavior.
