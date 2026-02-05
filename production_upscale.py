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
import yaml
import json
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm
from benchmarking import BenchmarkTracker

# Attempt imports for RealESRGAN
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    print("Error: Could not import RealESRGAN dependencies. Make sure you are in the correct environment.")
    sys.exit(1)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("upscale_production.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path: str, args: argparse.Namespace):
        self.config = self._load_config(config_path)
        self._override_with_args(args)
        self._validate_paths()

    def _load_config(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            logger.error(f"Config file not found: {path}")
            sys.exit(1)
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _override_with_args(self, args):
        # Override config with CLI args if provided
        if args.input_folder:
            self.config['io']['input_folder'] = args.input_folder
        if args.output_folder:
            self.config['io']['output_folder'] = args.output_folder
        if args.tile is not None:
             self.config['model']['tile_size'] = args.tile
        if args.scale is not None:
             self.config['model']['scale'] = args.scale

    def _validate_paths(self):
        # Ensure directories exist or create them
        os.makedirs(self.config['io']['output_folder'], exist_ok=True)
        if not os.path.exists(self.config['io']['input_folder']):
             logger.error(f"Input folder does not exist: {self.config['io']['input_folder']}")
             sys.exit(1)

    def get(self, key: str, default=None):
        # Helper for nested access could be added, but simple dict access is fine for now
        return self.config.get(key, default)

class BatchManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history_file = os.path.join(config['io']['output_folder'], config['io']['processed_history_file'])
        self.history = self._load_history()

    def _load_history(self) -> Dict[str, str]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("History file corrupted, starting fresh.")
                return {}
        return {}

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)

    def scan_input(self) -> list:
        input_folder = self.config['io']['input_folder']
        extensions = tuple(self.config['io']['extensions'])
        files = []
        for root, _, filenames in os.walk(input_folder):
            for filename in filenames:
                if filename.lower().endswith(extensions):
                    full_path = os.path.join(root, filename)
                    files.append(full_path)
        return files

    def is_processed(self, filepath: str) -> bool:
        # Check by filename (assuming unique filenames, or use relative path)
        # Using basename for simplicity, but relative path is safer for nested.
        # Let's use relative path from input folder.
        rel_path = os.path.relpath(filepath, self.config['io']['input_folder'])
        return rel_path in self.history and self.history[rel_path] == "completed"

    def mark_completed(self, filepath: str):
        rel_path = os.path.relpath(filepath, self.config['io']['input_folder'])
        self.history[rel_path] = "completed"
        self.save_history()
    
    def mark_failed(self, filepath: str, error: str):
        rel_path = os.path.relpath(filepath, self.config['io']['input_folder'])
        self.history[rel_path] = f"failed: {error}"
        self.save_history()

class UpscaleWorker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_model()

    def _init_model(self):
        model_conf = self.config['model']
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=model_conf['scale'])
        
        if not os.path.exists(model_conf['path']):
            logger.error(f"Model path not found: {model_conf['path']}")
            raise FileNotFoundError(f"Model not found at {model_conf['path']}")

        self.upsampler = RealESRGANer(
            scale=model_conf['scale'],
            model_path=model_conf['path'],
            model=self.model,
            tile=model_conf['tile_size'],
            tile_pad=model_conf['tile_pad'],
            pre_pad=model_conf['pre_pad'],
            half=model_conf['half_precision'],
            device=self.device
        )

    def _get_video_info(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return width, height, fps, count

    def process_video(self, input_path: str, output_path: str):
        logger.info(f"Processing: {input_path}")
        
        try:
            w, h, fps, total_frames = self._get_video_info(input_path)
            scale = self.config['model']['scale']
            target_w = w * scale
            target_h = h * scale
            
            # Prepare Queues
            batch_size = self.config['execution']['batch_size']
            read_queue = queue.Queue(maxsize=batch_size * 2)
            write_queue = queue.Queue(maxsize=batch_size * 2)
            
            # Start FFmpeg Reader
            reader_thread = threading.Thread(target=self._reader_worker, args=(input_path, w, h, read_queue), daemon=True)
            reader_thread.start()
            
            # Start FFmpeg Writer
            writer_thread = threading.Thread(target=self._writer_worker, args=(output_path, target_w, target_h, fps, write_queue), daemon=True)
            writer_thread.start()
            
            # Inference Loop
            self._inference_loop(read_queue, write_queue, total_frames, batch_size)
            
            # Finish
            reader_thread.join()
            writer_thread.join()
            logger.info(f"Finished: {output_path}")

        except Exception as e:
            logger.error(f"Failed processing {input_path}: {e}")
            raise e

    def _inference_loop(self, read_queue, write_queue, total_frames, batch_size):
        pbar = tqdm(total=total_frames, unit='frame', desc="Upscaling")
        batch = []
        
        while True:
            item = read_queue.get()
            if item is None:
                # Process remaining batch
                if batch:
                    for img in batch:
                        output, _ = self.upsampler.enhance(img, outscale=self.config['model']['scale'])
                        write_queue.put(output)
                    pbar.update(len(batch))
                break
            
            batch.append(item)
            if len(batch) >= batch_size:
                for img in batch:
                    output, _ = self.upsampler.enhance(img, outscale=self.config['model']['scale'])
                    write_queue.put(output)
                pbar.update(len(batch))
                batch = []
        
        write_queue.put(None) # Signal writer to finish
        pbar.close()

    def _reader_worker(self, path, w, h, read_queue):
        cmd = [
            'ffmpeg', '-y', '-i', path,
            '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
        ]
        # Hide ffmpeg output
        p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=10**8)
        
        frame_bytes = w * h * 3
        while True:
            raw = p.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            read_queue.put(img)
        read_queue.put(None)
        p.wait()

    def _writer_worker(self, path, w, h, fps, write_queue):
        ff_conf = self.config['ffmpeg']
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-',
            '-c:v', ff_conf['video_codec'], '-crf', str(ff_conf['crf']),
            '-preset', ff_conf['preset'], '-pix_fmt', ff_conf['output_pixel_format'],
            path
        ]
        p = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.DEVNULL)
        
        while True:
            frame = write_queue.get()
            if frame is None:
                break
            p.stdin.write(frame.tobytes())
            write_queue.task_done()
        p.stdin.close()
        p.wait()

def main():
    parser = argparse.ArgumentParser(description="Production Video Upscaling Pipeline")
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--input_folder', help='Override input folder')
    parser.add_argument('--output_folder', help='Override output folder')
    parser.add_argument('--tile', type=int, help='Override tile size')
    parser.add_argument('--scale', type=int, help='Override scale factor')
    args = parser.parse_args()

    # Load Configuration
    config_manager = ConfigManager(args.config, args)
    config = config_manager.config
    
    # Initialize Managers
    batch_manager = BatchManager(config)
    
    # Check if there are files to process
    files = batch_manager.scan_input()
    if not files:
        logger.warning(f"No files found in {config['io']['input_folder']} with extensions {config['io']['extensions']}")
        return

    logger.info(f"Found {len(files)} files to process.")
    
    # Initialize Worker (Load Model)
    try:
        worker = UpscaleWorker(config)
    except Exception as e:
        logger.critical(f"Failed to initialize worker: {e}")
        sys.exit(1)

    # Initialize Benchmark Tracker
    tracker = BenchmarkTracker(config['io']['output_folder'])
    tracker.start()
    processed_count = 0

    # Processing Loop
    for i, file_path in enumerate(files):
        if batch_manager.is_processed(file_path):
            logger.info(f"Skipping already processed: {file_path}")
            continue
        
        logger.info(f"[{i+1}/{len(files)}] Starting: {file_path}")
        
        # Determine output path
        rel_path = os.path.relpath(file_path, config['io']['input_folder'])
        base, _ = os.path.splitext(rel_path)
        output_filename = f"{base}_upscaled.mp4"
        output_path = os.path.join(config['io']['output_folder'], output_filename)
        
        # Ensure subdirectories exist in output if input was recursive
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            worker.process_video(file_path, output_path)
            batch_manager.mark_completed(file_path)
            processed_count += 1
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user.")
            tracker.stop() # Ensure we save even if interrupted
            tracker.save("benchmark_video_interrupted.json")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            batch_manager.mark_failed(file_path, str(e))
            # Continue to next file? Yes, usually for batch processing we want resilience.
            continue
            
    tracker.stop()
    tracker.add_custom_metric("processed_files", processed_count)
    tracker.add_custom_metric("config_model", config['model'])
    tracker.save("benchmark_video.json")

if __name__ == '__main__':
    main()
