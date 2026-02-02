import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:
        print(f"Downloaded {filename} successfully.")

def main():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    url = "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"
    filename = os.path.join(model_dir, "BSRGAN.pth")
    
    if not os.path.exists(filename):
        download_file(url, filename)
    else:
        print(f"{filename} already exists.")

if __name__ == "__main__":
    main()
