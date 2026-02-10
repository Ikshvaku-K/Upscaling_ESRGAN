# Understanding ONNX and TensorRT for Real-ESRGAN

This document provides a comprehensive explanation of how ONNX and TensorRT work together to dramatically accelerate the Real-ESRGAN video upscaling pipeline.

---

## The Problem: Why PyTorch is Slow for Inference

When you train a neural network in **PyTorch**, the framework is designed for flexibility:
- **Dynamic Computation Graphs**: PyTorch rebuilds the computational graph on every forward pass, allowing for easy debugging and experimentation.
- **Python Overhead**: Every operation involves Python interpreter calls, which add latency.
- **Generic Operations**: PyTorch uses general-purpose CUDA kernels that work across all GPUs but aren't optimized for your specific hardware.

This flexibility is great for research and training, but it's **inefficient for production inference** where you run the same model thousands of times on video frames.

---

## Part 1: ONNX (Open Neural Network Exchange)

### What is ONNX?
**ONNX** is an open-source format that represents neural networks as a **static computational graph**. Think of it as a blueprint or recipe that describes exactly what mathematical operations need to happen, in what order, without any Python code.

### The Analogy
Imagine you're baking a cake:
- **PyTorch** is like having a chef who reads the recipe out loud in real-time, decides what to do next, and occasionally improvises.
- **ONNX** is like having the recipe written down on paper—no interpretation needed, just follow the steps.

### What Happens During ONNX Export?

When you run [`export_onnx.py`](file:///home/isaslab-summer/Downloads/upscaling/export_onnx.py):

1. **Model Tracing**: PyTorch runs a dummy input (e.g., a 64×64 image tile) through the Real-ESRGAN model and records every operation.
2. **Graph Serialization**: The sequence of operations (convolutions, activations, upsampling) is saved as a `.onnx` file.
3. **Dynamic Axes**: The script specifies that height and width can vary (dynamic shapes), so the same `.onnx` file works for different tile sizes.

### Benefits of ONNX
- **Platform Independence**: The `.onnx` file can be loaded by TensorRT, ONNX Runtime, OpenVINO, or even run on mobile devices.
- **No Python Required**: The model is now a pure data file—no need for PyTorch to be installed during inference.
- **Optimization Ready**: The static graph makes it easier for downstream tools (like TensorRT) to analyze and optimize.

### Key Code in `export_onnx.py`
```python
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    opset_version=16,  # ONNX operator set version
    dynamic_axes={
        'input': {2: 'height', 3: 'width'},
        'output': {2: 'height', 3: 'width'}
    }
)
```

---

## Part 2: TensorRT (The GPU Compiler)

### What is TensorRT?
**TensorRT** is NVIDIA's SDK for high-performance deep learning inference. It takes the ONNX graph and **compiles it into a highly optimized execution engine** tailored specifically for your GPU architecture.

### The Analogy
Continuing the baking analogy:
- **ONNX** is the written recipe.
- **TensorRT** is a professional chef who reads the recipe once, then reorganizes the kitchen, pre-measures ingredients, and creates a streamlined assembly line specifically for your oven model.

### What Happens During TensorRT Conversion?

When you run [`trt_convert.py`](file:///home/isaslab-summer/Downloads/upscaling/src/upscaler/core/trt_convert.py):

1. **Layer Fusion**: TensorRT combines multiple operations into single GPU kernels. For example:
   - Instead of: `Conv → BatchNorm → ReLU` (3 separate GPU calls)
   - TensorRT does: `FusedConvBNReLU` (1 GPU call)
   - This reduces memory bandwidth and latency.

2. **Kernel Auto-Tuning**: TensorRT benchmarks different CUDA implementations of each operation and picks the fastest one for your specific GPU (e.g., RTX 4090 vs RTX 3060).

3. **Precision Calibration**: 
   - **FP32 (Full Precision)**: 32-bit floating point (default PyTorch).
   - **FP16 (Half Precision)**: 16-bit floating point—uses half the memory and runs ~2× faster on modern GPUs with Tensor Cores.
   - The `--fp16` flag in `trt_convert.py` enables this.

4. **Memory Optimization**: TensorRT pre-allocates GPU memory based on the optimization profiles you define (Min: 16×16, Opt: 512×512, Max: 640×640).

### Benefits of TensorRT
- **Massive Speedup**: Typically **2-5× faster** than PyTorch for the same model.
- **Lower Latency**: Reduced overhead means faster frame-to-frame processing.
- **Better GPU Utilization**: Fused operations keep the GPU busy instead of waiting for CPU to schedule the next operation.

### The Trade-Off: Hardware Lock-In
The `.trt` engine is **compiled for your specific GPU architecture**. If you:
- Upgrade from an RTX 3090 to an RTX 4090
- Update NVIDIA drivers significantly
- Move the engine to a different machine

You'll need to **re-run `trt_convert.py`** to rebuild the engine.

---

## Part 3: How This Affects Real-ESRGAN Performance

### Speed Comparison (Typical Results)

| Method | Time per 1080p Frame | Relative Speed |
|:-------|:---------------------|:---------------|
| PyTorch (FP32) | ~500ms | 1× (baseline) |
| ONNX Runtime | ~300ms | 1.7× faster |
| TensorRT (FP32) | ~200ms | 2.5× faster |
| TensorRT (FP16) | ~100ms | **5× faster** |

*Note: Actual performance depends on GPU model, tile size, and padding settings.*

### Memory Usage
- **PyTorch**: Stores intermediate activations in FP32, uses ~8GB VRAM for 1080p tiles.
- **TensorRT FP16**: Uses ~4GB VRAM for the same workload.

### Quality Impact
- **FP16 Precision**: The difference in upscaled image quality is **imperceptible** to the human eye. Neural networks are robust to small numerical precision changes.
- **No Accuracy Loss**: For Real-ESRGAN, FP16 produces visually identical results to FP32.

---

## Part 4: The Complete Pipeline

### Step-by-Step Workflow

```mermaid
graph LR
    A[PyTorch Model<br/>.pth] -->|export_onnx.py| B[ONNX Model<br/>.onnx]
    B -->|trt_convert.py| C[TensorRT Engine<br/>.trt]
    C -->|trt_run.py| D[Upscaled Video]
    
    style A fill:#ff6b6b
    style B fill:#4ecdc4
    style C fill:#45b7d1
    style D fill:#96ceb4
```

### 1. Export to ONNX
```bash
python export_onnx.py \
  --model models/RealESRGAN_x4plus.pth \
  --output models/realesrgan.onnx \
  --opset 16
```

**What happens:**
- Loads the PyTorch checkpoint
- Traces the model with a dummy 64×64 input
- Saves the static graph as `realesrgan.onnx`

### 2. Build TensorRT Engine
```bash
python src/upscaler/core/trt_convert.py \
  --onnx models/realesrgan.onnx \
  --output models/realesrgan.trt \
  --fp16 \
  --verbose
```

**What happens:**
- Parses the ONNX graph
- Optimizes for your GPU (e.g., RTX 4090)
- Enables FP16 precision
- Saves the compiled engine as `realesrgan.trt`

**Build time:** 5-15 minutes (one-time cost)

### 3. Run Inference
```bash
python src/upscaler/core/trt_run.py \
  --engine models/realesrgan.trt \
  --input video.mp4 \
  --output upscaled.mp4
```

**What happens:**
- Loads the pre-compiled TensorRT engine
- Processes video frames at maximum speed
- Outputs the upscaled video

---

## Part 5: Optimization Profiles Explained

In [`trt_convert.py`](file:///home/isaslab-summer/Downloads/upscaling/src/upscaler/core/trt_convert.py#L39), you'll see:

```python
profile.set_shape("input", 
    (1, 3, 16, 16),    # Min: tiny edge tiles
    (1, 3, 512, 512),  # Opt: standard tile size
    (1, 3, 640, 640)   # Max: tile + padding
)
```

### What This Means
- **Min (16×16)**: The smallest tile the engine will accept. Useful for edge cases where the video frame doesn't divide evenly.
- **Opt (512×512)**: The tile size TensorRT optimizes for. Most frames will use this size.
- **Max (640×640)**: The largest tile allowed. Accounts for padding (e.g., 512 + 10 pixels on each side).

### Why This Matters
TensorRT pre-allocates memory and optimizes kernel launches based on these profiles. If you try to pass a 1024×1024 tile, it will **fail** because it exceeds the max profile.

---

## Part 6: Common Questions

### Q: Can I skip ONNX and go straight to TensorRT?
**A:** No. TensorRT requires an ONNX model as input. The ONNX step is necessary to convert PyTorch's dynamic graph into a static format.

### Q: Do I need to rebuild the TensorRT engine every time?
**A:** No. Once built, the `.trt` file can be reused indefinitely—unless you:
- Change GPUs
- Update NVIDIA drivers significantly
- Modify the model architecture

### Q: What if I don't have an NVIDIA GPU?
**A:** TensorRT only works on NVIDIA GPUs. Alternatives:
- **ONNX Runtime**: Cross-platform, works on CPU/AMD/Intel GPUs (slower than TensorRT).
- **OpenVINO**: Optimized for Intel CPUs/GPUs.

### Q: Does FP16 reduce quality?
**A:** For Real-ESRGAN, the quality difference is **imperceptible**. Neural networks are designed to be robust to small numerical errors.

---

## Summary

| Stage | Input | Output | Purpose |
|:------|:------|:-------|:--------|
| **PyTorch** | Training data | `.pth` checkpoint | Train the model |
| **ONNX Export** | `.pth` | `.onnx` | Remove Python overhead, create static graph |
| **TensorRT Build** | `.onnx` | `.trt` | Compile for your GPU, enable FP16, fuse layers |
| **Inference** | `.trt` + video | Upscaled video | Run at maximum speed |

**Bottom Line:** ONNX and TensorRT transform Real-ESRGAN from a flexible research model into a production-ready, GPU-optimized inference engine that's **5× faster** with **half the memory usage**.
