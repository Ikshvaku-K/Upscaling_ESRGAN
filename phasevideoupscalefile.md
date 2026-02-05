# Phase 2 Video Upscaling Pipeline: Status & Objectives

## Current Status (Phase 1)
We have built a working **Proof of Concept (POC)** in `upscale_video.py`. It successfully runs the upscaler, but it is a "naive" implementation.

### 1. Extract frames with FFmpeg
*   **Status: [ ] Not Done**
*   **Current Approach**: Used `cv2.VideoCapture` to read frames one-by-one in Python.
*   **Why Change**: `ffmpeg` is significantly faster, more robust at decoding various formats (VFR, MKV, etc.), and efficient at piping raw integration.

### 2. Batch upscale frames
*   **Status: [ ] Not Done**
*   **Current Approach**: Process **one frame at a time** inside a simple `while` loop.
*   **Why Change**: Sending 1 frame to the GPU leaves it idle most of the time. By sending a batch (e.g., 4, 8, or 16 frames) at once, we saturate the CUDA cores and can achieve **2-4x speedups**.

### 3. Reassemble video (x265 / AV1)
*   **Status: [ ] Not Done**
*   **Current Approach**: Used `cv2.VideoWriter` with 'mp4v' (older MPEG-4).
*   **Why Change**: Modern codecs like **H.265 (HEVC)** or **AV1** compress 8K video much more efficiently. They maintain high quality with significantly lower file sizes. OpenCV's support for strictly controlling these encoding parameters is limited compared to direct `ffmpeg` usage.

### 4 & 5. Temporal Flicker & Stabilization
*   **Status: [ ] Observed**
*   **Current Issue**: Frame-wise upscaling causes "flickering" because the model processes every frame independently.
*   **Phase 2 Goal**: While true video stabilization requires video-specific networks (Video-SwinIR), using batching and consistent floating-point precision can help slightly. This remains a known limitation of frame-based GANs.

### 6 & 7. Encoding Tradeoffs & I/O Bottlenecks
*   **Status: [ ] Planned**
*   **Current Issue**: The CPU waits for the GPU, and the GPU waits for the CPU (Sequential I/O).
*   **Phase 2 Goal**: Implement a pipeline where:
    1.  **Thread 1**: Decodes video via FFmpeg -> buffer.
    2.  **Thread 2**: Batches frames -> GPU inference.
    3.  **Thread 3**: Encodes output -> FFmpeg (x265).
    *This parallelization minimizes bottlenecks.*

### 8. GPU vs CPU Workload Split
*   **Status: [ ] Optimization Needed**
*   **Goal**: Offload all decoding/encoding to hardware (if available) or efficient CPU threads, keeping the GPU 100% busy with Tensor math.

---

## Phase 2 Implementation Plan
We will create a new script `upscale_video_pipeline.py` that utilizes:
1.  **FFmpeg Piping**: Read raw video frames via standard input/output streams.
2.  **Dataset/DataLoader**: Use PyTorch constructs to handle batching efficiently.
3.  **Subprocess**: Call FFmpeg directly for the final encoding.
