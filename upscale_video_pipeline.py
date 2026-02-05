import cv2
import subprocess as sp
import numpy as np
import torch
import queue
import threading
import argparse
import os
import sys
import time
from tqdm import tqdm
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, count

def ffmpeg_reader_process(path):
    # Command to output raw video frames to pipe
    cmd = [
        'ffmpeg', '-y',
        '-i', path,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-'
    ]
    p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=10**8)
    return p

def ffmpeg_writer_process(path, width, height, fps, crf=20):
    # Command to read raw video frames from pipe and encode to x265
    # Input format must match what we write (bgr24)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-', 
        '-c:v', 'libx265', # Use x265 for efficiency
        '-crf', str(crf),
        '-preset', 'slow',
        '-pix_fmt', 'yuv420p', # compatible player format
        path
    ]
    p = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.DEVNULL)
    return p

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Video Upscaling Pipeline (FFmpeg + Batched I/O)")
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', help='Output video path')
    parser.add_argument('-s', '--scale', type=int, default=4, help='Scale factor')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size (0=auto)')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size (Works best with tile=0)')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16')
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if not args.output:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_upscaled.mp4"
    else:
        output_path = args.output

    # 1. Get Info
    print("Probing video...")
    w, h, fps, total_frames = get_video_info(input_path)
    print(f"Input: {w}x{h} @ {fps}fps, {total_frames} frames")
    
    target_w = w * args.scale
    target_h = h * args.scale
    print(f"Target: {target_w}x{target_h}")

    # 2. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model_path = "models/RealESRGAN_x4plus.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} missing")
        sys.exit(1)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=10,
        pre_pad=0,
        half=not args.fp32,
        device=device
    )

    # 3. Pipelines
    read_queue = queue.Queue(maxsize=args.batch_size * 2)
    write_queue = queue.Queue(maxsize=args.batch_size * 2)
    
    # Reader Thread
    def reader_worker():
        proc = ffmpeg_reader_process(input_path)
        frame_bytes = w * h * 3
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            # Convert to buffer immediately to free pipe
            img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            read_queue.put(img)
        read_queue.put(None) # Signal EOF
        proc.wait()

    # Writer Thread
    def writer_worker():
        proc = ffmpeg_writer_process(output_path, target_w, target_h, fps)
        while True:
            frame = write_queue.get()
            if frame is None:
                break
            proc.stdin.write(frame.tobytes())
            write_queue.task_done()
        proc.stdin.close()
        proc.wait()

    threading.Thread(target=reader_worker, daemon=True).start()
    threading.Thread(target=writer_worker, daemon=True).start()

    # 4. Processing Loop
    print(f"Processing (Batch Size: {args.batch_size})...")
    pbar = tqdm(total=total_frames)
    
    batch = []
    
    try:
        while True:
            # Collect Batch
            item = read_queue.get()
            
            if item is None:
                # Process remaining
                if batch:
                    # Serial fallback for safety if tile > 0 or complex
                    # Implementing True Batching with Tiling is extremely complex
                    # We will simply map over the batch for now, but parallel I/O is the gain here.
                    # If batch_size > 1 and tile=0, we could potentially stack.
                    # For this implementation, we stick to sequential inference on the GPU
                    # but decoupled from I/O waiting.
                    for img in batch:
                        output, _ = upsampler.enhance(img, outscale=args.scale)
                        write_queue.put(output)
                    pbar.update(len(batch))
                break
            
            batch.append(item)
            
            if len(batch) >= args.batch_size:
                # Inference
                for img in batch:
                    output, _ = upsampler.enhance(img, outscale=args.scale)
                    write_queue.put(output)
                
                pbar.update(len(batch))
                batch = []

        # Wait for writer
        write_queue.put(None)
        write_queue.join()
        pbar.close()
        print("\nComplete!")

    except KeyboardInterrupt:
        print("Interrupted")
    
if __name__ == '__main__':
    main()
