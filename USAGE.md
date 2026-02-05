# Upscaler Package Usage Guide

This package (`upscaler`) provides a comprehensive command-line interface for AI-based image and video upscaling, as well as hardware optimization and TensorRT engine conversion.

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA drivers installed (for GPU acceleration)
- FFmpeg (for video processing)

### Install
To install the package in editable mode (recommended for development):
```bash
pip install -e .
```

To install as a standard package:
```bash
pip install .
```

## CLI Commands

The main command is `upscaler`.

```bash
upscaler --help
```

### 1. Hardware Optimization (`optimize`)

Check your hardware capabilities and get recommended settings (Batch Size, Tile Size, FP16 support).

```bash
upscaler optimize
```

### 2. Video Upscaling (`video`)

Upscale videos using the production pipeline.

```bash
upscaler video --input_folder inputs/ --output_folder outputs/
```

**Configuration:**
The command uses a default configuration, but you can override specific settings via CLI flags or provide a custom `config.yaml`.

**CLI Options:**
- `--config`: Path to YAML configuration file (optional).
- `--input_folder`: Override input directory (default: `inputs`).
- `--output_folder`: Override output directory (default: `outputs`).
- `--scale`: Override upscaling factor (default: 4).
- `--tile`: Override tile size (default: 0/auto).

**Custom Config File (`config.yaml`):**
If you need fine-grained control (e.g., FFmpeg encoding settings), create a `config.yaml`:
```yaml
model:
  name: "RealESRGAN_x4plus"
  path: "models/RealESRGAN_x4plus.pth"
  scale: 4
  tile_size: 0        # 0 for auto
  half_precision: true

io:
  input_folder: "inputs"
  output_folder: "outputs"
  extensions: [".mp4", ".mov", ".avi", ".mkv"]

ffmpeg:
  crf: 20
  preset: "slow"
  video_codec: "libx265"

execution:
  batch_size: 1
```

### 3. Image Upscaling (`image`)

Upscale a single image.

```bash
upscaler image input.jpg -o output.png -s 4 --fp32
```

**Options:**
- `-n, --model_name`: Model name (default: `RealESRGAN_x4plus`).
- `-o, --output`: Output file path.
- `-s, --outscale`: Upscaling factor.
- `-t, --tile`: Tile size (0 for no tiling).
- `--fp32`: Use FP32 precision (default is half precision if supported).
- `--gpu-id`: Specific GPU ID to use.

### 4. TensorRT Conversion (`convert-trt`)

Convert an ONNX model to a TensorRT engine for faster inference.

```bash
upscaler convert-trt --onnx models/realesrgan.onnx --output models/realesrgan.trt --fp16
```

**Options:**
- `--onnx`: Path to input ONNX model.
- `--output`: Path to output TensorRT engine.
- `--fp16`: Enable FP16 precision (recommended for newer GPUs).
- `--verbose`: Enable verbose logging.
