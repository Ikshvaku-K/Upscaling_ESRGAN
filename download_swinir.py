import hashlib
import os
import sys

import requests
from tqdm import tqdm

# SHA-256 of the official release asset (JingyunLiang/SwinIR v0.0). Model
# weights are loaded with torch.load (pickle), so integrity must be verified
# before use.
EXPECTED_SHA256 = "4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215"

def sha256_of(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def download_file(url, filename, expected_sha256=None):
    print(f"Downloading {filename}...")
    # Download to a temp file first so an interrupted run never leaves a
    # partial file at the final path (which would be skipped on retry).
    tmp_path = filename + ".part"
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024 * 64
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(tmp_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                raise IOError(f"Incomplete download: got {progress_bar.n} of {total_size_in_bytes} bytes")

        if expected_sha256:
            actual = sha256_of(tmp_path)
            if actual != expected_sha256:
                raise IOError(
                    f"Checksum mismatch for {filename}:\n"
                    f"  expected {expected_sha256}\n"
                    f"  got      {actual}\n"
                    "The file was NOT installed. It may be corrupted or tampered with."
                )
            print("Checksum verified (SHA-256 OK).")

        os.replace(tmp_path, filename)
        print(f"Downloaded {filename} successfully.")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def main():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    filename = os.path.join(model_dir, "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")

    if not os.path.exists(filename):
        try:
            download_file(url, filename, EXPECTED_SHA256)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        print(f"{filename} already exists.")

if __name__ == "__main__":
    main()
