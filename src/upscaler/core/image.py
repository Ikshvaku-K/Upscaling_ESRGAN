import cv2
import torch
import argparse
import os
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def main():
    parser = argparse.ArgumentParser(description="Upscale an image to 8K using Real-ESRGAN.")
    parser.add_argument('input', nargs='?', default='input_4k.jpg', help='Path to input image')
    parser.add_argument('-o', '--output', help='Path to output image (optional)')
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    # Determine output path if not provided
    if args.output:
        output_path = args.output
    else:
        # Default: filename_8k.png
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_8k{ext}"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model architecture
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    # Load ESRGAN
    upsampler = RealESRGANer(
        scale=4,
        model_path="models/RealESRGAN_x4plus.pth",
        model=model,
        tile=512,          # VRAM-safe
        tile_pad=10,
        pre_pad=0,
        half=True,         # FP16
        device=device
    )

    # Load image
    print(f"Loading {input_path}...")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image '{input_path}'.")
        return

    # Upscale
    print("Upscaling (this may take a moment)...")
    try:
        output, _ = upsampler.enhance(img, outscale=2)
        
        # Save 8K result
        cv2.imwrite(output_path, output)
        print(f" Upscale complete: {output_path}")
        
    except Exception as e:
        print(f"Error during upscaling: {e}")

if __name__ == "__main__":
    main()
