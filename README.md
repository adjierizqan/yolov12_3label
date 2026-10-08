# YOLOv12 experiments

An experimental object detection workspace based on the [YOLOv12 project](https://github.com/sunsmarterjie/yolov12). It includes a local model codebase, example files and a Gradio interface for image and video inference.

## Repository contents

- `app.py`: Gradio interface with model, image size and confidence controls
- `ultralytics/`: model and inference code
- `examples/`: sample integrations and notebooks
- `tests/`: test modules
- `requirements.txt`: dependency list

The interface expects YOLOv12 model weights such as `yolov12n.pt` or `yolov12s.pt`. The dependency list includes a Linux/CUDA-specific FlashAttention wheel, so it is not a portable one-command installation on every machine.

## Status

This is a research workspace, not a packaged application. No independently verified benchmark or deployment claim is made here. See the upstream repository and the included `LICENSE` for the underlying project and licensing context.
