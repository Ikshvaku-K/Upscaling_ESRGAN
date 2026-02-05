# AI Image Upscaling Project (Real-ESRGAN, SwinIR, BSRGAN)

This repository contains a suite of tools and scripts for upscaling images to 8K resolution using state-of-the-art super-resolution models. The project was developed in phases, each focusing on different models, benchmarking, and quality comparisons.

---

## Project Structure & Phases

### **Phase 1: Basic Upscaling (`upscale_image.py`)**
*   **Goal**: Create a simple, robust command-line tool to upscale *any* image to 8K.
*   **Core Script**: `upscale_image.py`
*   **Usage**:
    ```bash
    python upscale_image.py input.jpg -o output_8k.png
    ```
*   **Default Model**: **Real-ESRGAN** (specifically `RealESRGAN_x4plus.pth`).
*   **Key Features**:
    *   **Tiling**: Splits large images into smaller tiles (patches) to avoid running out of GPU memory (VRAM).
    *   **Tile Padding**: Adds overlap between tiles to prevent visible seams (grid lines) in the final image.

### **Phase 2: Benchmarking (`benchmark_phase3.py`)**
*   **Goal**: Measure the raw performance of the upscaling process.
*   **Core Script**: `benchmark_phase3.py`
*   **Metrics Tracked**:
    *   **Execution Time**: How long the upscaling takes in seconds.
    *   **Peak Memory Usage**: The maximum amount of GPU memory (VRAM) consumed.
*   **Outcome**: Established a baseline for how fast Real-ESRGAN runs on the current hardware.

### **Phase 3: Model Comparison (`compare_models.py`)**
*   **Goal**: Compare the quality and speed of different models (Real-ESRGAN vs. SwinIR vs. BSRGAN).
*   **Core Script**: `compare_models.py`
*   **Workflow**:
    1.  Create a 512x512 center crop of the input image.
    2.  Run each model on the same crop.
    3.  Compute quality metrics (Color Drift).
    4.  Generate difference maps ("diff maps") to visualize changes.
*   **Artifacts Generated**:
    *   `model_comparison_result.jpg`: Side-by-side visual comparison.
    *   `artifact_analysis.md`: Detailed report of the findings.

---

## Models Used & Features

| Model Name | Type | Best Used For | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Real-ESRGAN** | **GAN** (Generative Adversarial Network) | Batch processing, Animation, Sketches | **Fastest** (~3-4x faster than SwinIR), Very sharp edges. | **Hallucinations** (invents details), **Ringing** artifacts, High **Color Drift** (changes colors slightly). |
| **SwinIR** | **Transformer** (Swin Transformer) | Photography, Art, Archival | **Best Quality**, Low Color Drift (accurate colors), Natural textures. | **Slow**, Uses more VRAM. |
| **BSRGAN** | **GAN** | General Restoration | Designed for heavy degradation (JPEG artifacts, noise). | Compatibility issues in some environments (Failed in our test). |

---

## Dictionary of Key Terms

### **Upscaling / Super-Resolution (SR)**
The process of increasing the resolution of an image (e.g., 4K → 8K) while attempting to reconstruct missing details. AI models "guess" or "hallucinate" these details based on training data.

### **GAN (Generative Adversarial Network)**
A type of AI model where two networks compete: a *Generator* creates an image, and a *Discriminator* tries to spot if it's fake. This competition forces the Generator to produce very sharp, realistic-looking images, but they can sometimes look "too sharp" or artificial.

### **Transformer (Vision Transformer)**
A newer type of AI architecture originally designed for text (like GPT) but adapted for images. SwinIR uses this. It is excellent at understanding long-range dependencies in an image (e.g., repeating textures), leading to more consistent and natural results than GANs.

### **Inference Time**
The time it takes for the AI model to process an image and produce the result. Lower is better.

### **VRAM (Video RAM)**
The memory on your Graphics Card (GPU). High-resolution upscaling requires a lot of VRAM. If you run out, you get an **OOM (Out Of Memory)** error. Tiling helps reduce VRAM usage.

### **Color Drift / Delta E**
*   **Color Drift**: When the upscaled image has slightly different colors than the original. Ideally, you want 0 drift.
*   **Delta E**: A metric to measure color difference.
    *   **< 1.0**: Imperceptible.
    *   **1.0 - 2.0**: Barely noticeable (Valid for high-end work).
    *   **> 2.3**: JND (Just Noticeable Difference).
    *   **Real-ESRGAN** had a drift of ~1.45 (Noticeable).
    *   **SwinIR** had a drift of ~0.38 (Imperceptible).

### **Ringing / Halo Artifacts**
Visual distortions that look like "echoes" or "ghost lines" around sharp edges. This happens when a model over-sharpens an image. Real-ESRGAN is prone to this.

### **Hallucination**
When the AI invents details that do not exist in the original image. For example, turning a blurry patch of grass into a specific leaf pattern that isn't actually there. Useful for artistic upscaling, bad for forensic or faithful restoration.

### **Tile Padding**
When processing large images in tiles (blocks), the edges of each tile can look weird because the model doesn't see the neighboring pixels. **Padding** adds extra overlapping pixels around each tile so the model can "see" the context, ensuring seamless merging. Without padding, you get visible grid lines.

---

## Setup & Requirements

1.  **Virtual Environment**: Ensure you are running in the `venv` provided.
    ```bash
    source venv/bin/activate
    ```
2.  **Dependencies**:
    *   `torch` (PyTorch) for model execution.
    *   `basicsr` / `realesrgan` for model architectures.
    *   `cv2` (OpenCV) for image handling.
    *   **FFmpeg** (System Requirement): Must be installed for Phase 2 Video Pipeline.
        *   `sudo apt install ffmpeg`

### **Phase 2: Video Pipeline (`upscale_video_pipeline.py`)**
*   **Goal**: Professional video upscaling with high efficiency.
*   **Core Script**: `upscale_video_pipeline.py`
*   **Features**:
    *   **FFmpeg Integration**: Fast, robust video decoding/encoding.
    *   **Batched I/O**: Reads/writes in parallel threads to keep GPU busy.
    *   **H.265 (HEVC)**: Outputs high-efficiency video.
*   **Usage**:
    ```bash
    python upscale_video_pipeline.py input.mp4 -o output.mp4 -b 4
    ```

### **Phase 4: Extreme Performance (TensorRT)**
*   **Goal**: Extreme performance via TensorRT and hardware-aware optimization.
*   **Core Scripts**: `optimize_hardware_config.py`, `convert_trt_engine.py`, `upscale_pipeline_trt.py`.
*   **Key Features**:
    *   **Hardware Analysis**: Automatically detects GPU VRAM and Compute Capability to suggest optimal settings.
    *   **TensorRT Engine**: Converts PyTorch models to optimized Engines (FP32/FP16) for 2-5x speedup.
    *   **Tiled Inference**: Handles 4K/8K inputs on limited VRAM by splitting images into tiles.

#### **Step 1: hardware Optimization**
Before running heavy upscaling, check your hardware capabilities and get recommended settings (Batch Size, Tile Size, FP16 support).
```bash
python optimize_hardware_config.py
```
*Output Example:*
```json
{
    "batch_size": 2,
    "tile_size": 512,
    "fp16": true,
    "trtexec_workspace": 4096
}
```

#### **Step 2: Build TensorRT Engine**
Convert the PyTorch model (`.pth` -> `.onnx` -> `.trt`).
*   **Standard Build (FP32)**:
    ```bash
    python convert_trt_engine.py --onnx models/realesrgan.onnx --output models/realesrgan.trt
    ```
*   **High Performance (FP16)** (Requires Nvidia Volta/Turing or newer):
    ```bash
    python convert_trt_engine.py --onnx models/realesrgan.onnx --output models/realesrgan_fp16.trt --fp16
    ```

#### **Step 3: Run Inference**
Upscale a video using the generated engine. Use the `tile_size` suggested by Step 1.
```bash
python upscale_pipeline_trt.py input.mp4 --engine models/realesrgan_fp16.trt --output output.mp4 --tile 512
```
*   `--tile`: Size of chunks to process. Lower this (e.g., 256) if you run out of VRAM.

