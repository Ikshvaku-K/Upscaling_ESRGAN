import argparse
import sys
from upscaler.core.image import main as image_main
from upscaler.core.video import main as video_main
from upscaler.core.trt_convert import build_engine
from upscaler.utils.hardware import suggest_settings, get_gpu_info

def optimize_main(args):
    info = get_gpu_info()
    suggest_settings(info)

def convert_main(args):
    build_engine(args.onnx, args.output, args.fp16, args.verbose)

def main():
    parser = argparse.ArgumentParser(description="Upscaler CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Video Upscale
    video_parser = subparsers.add_parser("video", help="Upscale video")
    video_parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    video_parser.add_argument('--input_folder', help='Override input folder')
    video_parser.add_argument('--output_folder', help='Override output folder')
    video_parser.add_argument('--tile', type=int, help='Override tile size')
    video_parser.add_argument('--scale', type=int, help='Override scale factor')

    # Video Upscale
    image_parser = subparsers.add_parser("image", help="Upscale image")
    image_parser.add_argument('input', help='Input image path')
    image_parser.add_argument('-n', '--model_name', default='RealESRGAN_x4plus', help='Model name')
    image_parser.add_argument('-o', '--output', default='output.png', help='Output path')
    image_parser.add_argument('-s', '--outscale', type=float, default=4, help='Upscaling factor')
    image_parser.add_argument('--suffix', type=str, default='out', help='Suffix for output')
    image_parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size')
    image_parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    image_parser.add_argument('--pre_pad', type=int, default=0, help='Pre-padding')
    image_parser.add_argument('--fp32', action='store_true', help='Use FP32')
    image_parser.add_argument('--gpu-id', type=int, default=None, help='GPU ID')

    # Hardware Optimization
    opt_parser = subparsers.add_parser("optimize", help="Check hardware and suggest settings")

    # TensorRT Conversion
    trt_parser = subparsers.add_parser("convert-trt", help="Convert model to TensorRT")
    trt_parser.add_argument('--onnx', required=True, help='Path to .onnx model')
    trt_parser.add_argument('--output', required=True, help='Path to output .trt engine')
    trt_parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    trt_parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.command == "video":
        # Hack to pass args to video_main which uses argparse internally
        sys.argv = [sys.argv[0]] 
        if args.config: sys.argv.extend(['--config', args.config])
        if args.input_folder: sys.argv.extend(['--input_folder', args.input_folder])
        if args.output_folder: sys.argv.extend(['--output_folder', args.output_folder])
        if args.tile: sys.argv.extend(['--tile', str(args.tile)])
        if args.scale: sys.argv.extend(['--scale', str(args.scale)])
        video_main()
    elif args.command == "image":
         # Similar hack for image_main
        sys.argv = [sys.argv[0], args.input, '-n', args.model_name, '-o', args.output, 
                    '-s', str(args.outscale), '--suffix', args.suffix, '-t', str(args.tile), 
                    '--tile_pad', str(args.tile_pad), '--pre_pad', str(args.pre_pad)]
        if args.fp32: sys.argv.append('--fp32')
        if args.gpu_id is not None: sys.argv.extend(['--gpu-id', str(args.gpu_id)])
        image_main()
    elif args.command == "optimize":
        optimize_main(args)
    elif args.command == "convert-trt":
        convert_main(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
