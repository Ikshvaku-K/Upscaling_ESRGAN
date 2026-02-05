import argparse
import json
import sys
import os
from beautifultable import BeautifulTable

def load_benchmark(path):
    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        return json.load(f)

def compare_benchmarks(files):
    table = BeautifulTable()
    table.columns.header = ["Metric", "Baseline", "Candidate", "Diff %"]
    
    if len(files) < 2:
        print("Error: Need at least 2 files to compare.")
        sys.exit(1)
        
    base = load_benchmark(files[0])
    curr = load_benchmark(files[1])
    
    metrics = [
        ("Runtime (s)", "runtime_seconds"),
        ("Avg Power (W)", ["gpu", "avg_power_watts"]),
        ("Peak VRAM (MB)", ["gpu", "peak_vram_mb"]),
    ]
    
    # Check for custom metrics like sec_per_image
    if "sec_per_image" in base and "sec_per_image" in curr:
        metrics.append(("Sec/Image", "sec_per_image"))
    if "processed_files" in base and "processed_files" in curr:
         # normalizing runtime by files might be better if counts differ
         pass

    print(f"Comparing:\n  Baseline: {files[0]}\n  Candidate: {files[1]}\n")

    for label, key in metrics:
        val1 = get_nested(base, key)
        val2 = get_nested(curr, key)
        
        if val1 is None or val2 is None:
            continue
            
        diff = ((val2 - val1) / val1) * 100 if val1 != 0 else 0.0
        row = [label, f"{val1:.2f}", f"{val2:.2f}", f"{diff:+.2f}%"]
        
        # Colorize diff? (CLI only)
        # For markdown output we just print the table
        table.rows.append(row)

    print(table)
    
    # Check for regression thresolds?
    # e.g. if runtime increased by > 10%
    
def get_nested(data, key):
    if isinstance(key, list):
        for k in key:
            data = data.get(k, {})
        return data if not isinstance(data, dict) else None
    return data.get(key)

def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark JSON files.")
    parser.add_argument('files', nargs='+', help='Path to benchmark JSON files (Baseline, Candidate)')
    args = parser.parse_args()
    
    compare_benchmarks(args.files)

if __name__ == '__main__':
    main()
