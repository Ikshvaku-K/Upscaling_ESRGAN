import cv2
import torch
import argparse
import os
import sys
from tqdm import tqdm
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def main():
    parser = argparse.ArgumentParser(description="Upscale a video using Real-ESRGAN.")
    parser.add_argument('input', help='Path to input video')
    parser.add_argument('-o', '--output', help='Path to output video (optional)')
    parser.add_argument('-s', '--scale', type=int, default=4, help='Upsaling factor (default: 4)')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size (0 for auto/no tile). Use smaller value (e.g. 400) if low VRAM')
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_upscaled.mp4"

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Model Architecture
    # Currently hardcoded for RealESRGAN_x4plus which uses RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    model_path = "models/RealESRGAN_x4plus.pth"
    if not os.path.exists(model_path):
        # Fallback or try to find it in default location if installed as package? 
        # For this environment, we assume models are in 'models/' as seen in list_dir
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    # Upsampler
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=10,
        pre_pad=0,
        half=True, # FP16
        device=device
    )

    # Video Capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_path}'")
        sys.exit(1)

    # Video Properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input Video: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    upscale_factor = args.scale # Note: The model is x4, but we can outscale to something else if needed, 
                                # but usually we match model scale for best quality.
                                # RealESRGANer.enhance 'outscale' param controls final resize.
                                # If args.scale != 4, RealESRGANer will resize result.

    output_width = int(width * args.scale)
    output_height = int(height * args.scale)
    print(f"Target Resolution: {output_width}x{output_height}")

    # Video Writer
    # using mp4v codec for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    print(f"Processing... Output will be saved to {output_path}")

    try:
        if total_frames > 0:
            pbar = tqdm(total=total_frames, unit='frame')
        else:
            pbar = tqdm(unit='frame')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Upscale
            try:
                # outscale argument determines the final output size relative to input
                output_frame, _ = upsampler.enhance(frame, outscale=args.scale)
                out.write(output_frame)
            except RuntimeError as e:
                print(f"\nRuntime Error during processing: {e}")
                print("Try using a smaller tile size with --tile argument (e.g., -t 400)")
                break
            
            pbar.update(1)

        pbar.close()
        print("\nVideo upscaling complete.")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Saved: {output_path}")
    print("Note: Audio was not processed (requires ffmpeg integration).")

if __name__ == "__main__":
    main()
