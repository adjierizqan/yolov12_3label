# YOLOv12 Three-Label Research Workspace

Experimental computer-vision repository based on the YOLOv12 and Ultralytics
codebases. The main project-specific addition is a `timm`-backed Swin-Tiny
feature extractor that can be used as a multi-scale backbone for a YOLOv12
detection head.

This is research code rather than a published benchmark. Training metrics in
`logs/` are retained as experiment records; no accuracy claim should be assumed
without the matching dataset split, checkpoint, and evaluation command.

## Project-specific components

| Path | Purpose |
| --- | --- |
| `ultralytics/nn/modules/swin_backbone.py` | Swin-Tiny wrapper returning P3/P4/P5 feature maps |
| `ultralytics/cfg/models/v12/yolov12_swin_t.yaml` | Example YOLOv12 detection head using the Swin features |
| `ultralytics/nn/tasks.py` | Model parser registration for the custom backbone |
| `app.py` | Gradio interface for image and video inference |

## Environment

The pinned environment targets Python 3.11, PyTorch 2.2, and a CUDA-capable
Linux host.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

FlashAttention is hardware- and PyTorch-specific, so it is not installed from
a local wheel path. Install a build compatible with your CUDA/PyTorch stack,
for example:

```bash
pip install flash-attn --no-build-isolation
```

## Swin-backed training

Edit `nc` in the model YAML and provide a dataset YAML whose class list matches
the experiment. The checked-in example defaults to a single class.

```bash
yolo detect train \
  model=ultralytics/cfg/models/v12/yolov12_swin_t.yaml \
  data=/path/to/dataset.yaml \
  epochs=100 \
  imgsz=640 \
  device=0
```

Pretrained Swin weights are downloaded by `timm` on first use. Set
`pretrained: false` in the integration code when running in an offline
environment.

## Inference interface

```bash
python app.py
```

The Gradio app supports image and video input and expects a compatible YOLOv12
checkpoint such as `yolov12s.pt`.

## Validation

The lightweight GitHub Actions workflow checks Python syntax. The inherited
Ultralytics test suite includes network-, model-, and GPU-dependent tests; run
the relevant subset in the target CUDA environment before comparing models.

## Upstream and license

- YOLOv12 paper and implementation: <https://github.com/sunsmarterjie/yolov12>
- Ultralytics framework: <https://github.com/ultralytics/ultralytics>

The repository retains the upstream AGPL-3.0 license. Review that license before
redistribution or deployment.
