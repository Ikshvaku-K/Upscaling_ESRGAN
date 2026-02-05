import cv2
import torch
import argparse
import os
import sys
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
        logging.FileHandler("upscale_production_images.log"),
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
            self.config['images']['input_folder'] = args.input_folder
        if args.output_folder:
            self.config['images']['output_folder'] = args.output_folder
        if args.tile is not None:
             self.config['model']['tile_size'] = args.tile
        if args.scale is not None:
             self.config['model']['scale'] = args.scale

    def _validate_paths(self):
        # Ensure directories exist or create them
        os.makedirs(self.config['images']['output_folder'], exist_ok=True)
        if not os.path.exists(self.config['images']['input_folder']):
             # If input folder doesn't exist, just warn? Or create if user wants?
             # Better to error if user expects input.
             # Check if we are running image mode specifically? 
             # For now, we assume this script assumes the folders exist or are passed.
             # We won't exit hard here if using default config but no folder, 
             # but usually for batch processing we want the folder.
             if not os.path.exists(self.config['images']['input_folder']):
                  logger.warning(f"Input folder not found: {self.config['images']['input_folder']}")

class ImageBatchManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history_file = os.path.join(config['images']['output_folder'], config['images']['processed_history_file'])
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
        input_folder = self.config['images']['input_folder']
        if not os.path.exists(input_folder):
            return []
            
        extensions = tuple(self.config['images']['extensions'])
        files = []
        for root, _, filenames in os.walk(input_folder):
            for filename in filenames:
                if filename.lower().endswith(extensions):
                    full_path = os.path.join(root, filename)
                    files.append(full_path)
        return files

    def is_processed(self, filepath: str) -> bool:
        rel_path = os.path.relpath(filepath, self.config['images']['input_folder'])
        return rel_path in self.history and self.history[rel_path] == "completed"

    def mark_completed(self, filepath: str):
        rel_path = os.path.relpath(filepath, self.config['images']['input_folder'])
        self.history[rel_path] = "completed"
        self.save_history()
    
    def mark_failed(self, filepath: str, error: str):
        rel_path = os.path.relpath(filepath, self.config['images']['input_folder'])
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

    def process_image(self, input_path: str, output_path: str):
        logger.info(f"Processing: {input_path}")
        
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")

        try:
            output, _ = self.upsampler.enhance(img, outscale=self.config['model']['scale'])
            cv2.imwrite(output_path, output)
            logger.info(f"Saved: {output_path}")

        except Exception as e:
            logger.error(f"Failed processing {input_path}: {e}")
            raise e

def main():
    parser = argparse.ArgumentParser(description="Production Image Upscaling Pipeline")
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
    batch_manager = ImageBatchManager(config)
    
    # Check if there are files to process
    files = batch_manager.scan_input()
    if not files:
        if not os.path.exists(config['images']['input_folder']):
             logger.error(f"Input folder '{config['images']['input_folder']}' does not exist.")
        else:
             logger.warning(f"No files found in {config['images']['input_folder']} with extensions {config['images']['extensions']}")
        return

    logger.info(f"Found {len(files)} images to process.")
    
    # Initialize Worker
    try:
        worker = UpscaleWorker(config)
    except Exception as e:
        logger.critical(f"Failed to initialize worker: {e}")
        sys.exit(1)

    # Initialize Benchmark Tracker
    tracker = BenchmarkTracker(config['images']['output_folder'])
    tracker.start()
    processed_count = 0

    # Processing Loop
    for i, file_path in enumerate(tqdm(files, desc="Batch Progress")):
        if batch_manager.is_processed(file_path):
            # logger.info(f"Skipping already processed: {file_path}") # lessen noise in tqdm
            continue
        
        # logger.info(f"[{i+1}/{len(files)}] Starting: {file_path}")
        
        # Determine output path
        rel_path = os.path.relpath(file_path, config['images']['input_folder'])
        base, ext = os.path.splitext(rel_path)
        output_filename = f"{base}_upscaled.png" # Force PNG for lossless quality often preferred in upscaling
        output_path = os.path.join(config['images']['output_folder'], output_filename)
        
        # Ensure subdirectories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            worker.process_image(file_path, output_path)
            batch_manager.mark_completed(file_path)
            processed_count += 1
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user.")
            tracker.stop()
            tracker.save("benchmark_image_interrupted.json")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            batch_manager.mark_failed(file_path, str(e))
            continue
            
    tracker.stop()
    tracker.add_custom_metric("processed_images", processed_count)
    if processed_count > 0:
        tracker.add_custom_metric("sec_per_image", tracker.stats["runtime_seconds"] / processed_count)
    tracker.add_custom_metric("config_model", config['model'])
    tracker.save("benchmark_image.json")

if __name__ == '__main__':
    main()
