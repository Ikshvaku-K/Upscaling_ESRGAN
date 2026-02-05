import time
import threading
import subprocess
import json
import os
import torch
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class BenchmarkTracker:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.stats = {
            "runtime_seconds": 0.0,
            "gpu": {
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "peak_vram_mb": 0.0,
                "avg_power_watts": 0.0,
                "power_samples": []
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._running = False
        self._power_thread = None

    def start(self):
        self._running = True
        self.start_time = time.time()
        self._reset_cuda_stats()
        
        # Start power monitoring thread if GPU is available
        if torch.cuda.is_available():
            self._power_thread = threading.Thread(target=self._monitor_power, daemon=True)
            self._power_thread.start()

    def stop(self):
        self._running = False
        self.end_time = time.time()
        self.stats["runtime_seconds"] = self.end_time - self.start_time
        
        if self._power_thread:
            self._power_thread.join(timeout=1.0)
            
        self._capture_vram_peaks()
        self._finalize_power_stats()

    def _reset_cuda_stats(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

    def _monitor_power(self):
        """Polls nvidia-smi for power draw."""
        while self._running:
            try:
                # nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    power = float(result.stdout.strip())
                    self.stats["gpu"]["power_samples"].append(power)
            except Exception as e:
                # Silently fail for power monitoring to not interrupt main flow
                pass
            time.sleep(1.0) # Poll every second

    def _capture_vram_peaks(self):
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            self.stats["gpu"]["peak_vram_mb"] = peak_bytes / (1024 * 1024)

    def _finalize_power_stats(self):
        samples = self.stats["gpu"]["power_samples"]
        if samples:
            self.stats["gpu"]["avg_power_watts"] = sum(samples) / len(samples)
            # Optional: remove raw samples to keep JSON clean? Or keep for charts?
            # Keeping samples allows for plotting later if needed.
        else:
            self.stats["gpu"]["avg_power_watts"] = 0.0

    def add_custom_metric(self, name: str, value: Any):
        self.stats[name] = value

    def save(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, 'w') as f:
                json.dump(self.stats, f, indent=4)
            logger.info(f"Benchmark report saved to: {path}")
        except IOError as e:
            logger.error(f"Failed to save benchmark report: {e}")
