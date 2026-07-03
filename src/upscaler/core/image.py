import cv2
import torch
import argparse
import os
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def main(argv=None):
    parser = argparse.ArgumentParser(description="Upscale an image using Real-ESRGAN.")
    parser.add_argument('input', nargs='?', default='input_4k.jpg', help='Path to input image')
    parser.add_argument('-n', '--model_name', default='RealESRGAN_x4plus', help='Model name')
    parser.add_argument('-o', '--output', help='Path to output image (optional)')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='Final upscaling factor')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix appended to the input filename when no output path is given')
    parser.add_argument('-t', '--tile', type=int, default=512, help='Tile size (0 disables tiling)')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre-padding')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 precision (default FP16 on GPU)')
    parser.add_argument('--gpu-id', type=int, default=None, help='GPU ID')
    args = parser.parse_args(argv)

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return 1

    # Determine output path if not provided
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_{args.suffix}{ext}"

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Half precision is only supported on CUDA; on CPU it is unsupported/very slow
    use_half = (not args.fp32) and device.type == 'cuda'

    # Model architecture
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    model_path = os.path.join("models", f"{args.model_name}.pth")
    if not os.path.isfile(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return 1

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=use_half,
        device=device
    )

    # Load image
    print(f"Loading {input_path}...")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image '{input_path}'.")
        return 1

    # Upscale
    print("Upscaling (this may take a moment)...")
    try:
        output, _ = upsampler.enhance(img, outscale=args.outscale)

        if not cv2.imwrite(output_path, output):
            print(f"Error: Could not write output to '{output_path}'.")
            return 1
        print(f"Upscale complete: {output_path}")
        return 0

    except Exception as e:
        print(f"Error during upscaling: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
