import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet
import argparse
import os
import sys

def export_onnx(model_path, output_path, opset=13):
    print(f"Loading model from {model_path}...")
    # RealESRGAN x4 plus parameters
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    if hasattr(model, 'load_state_dict'):
        try:
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
            else:
                keyname = 'params'
            model.load_state_dict(loadnet[keyname], strict=True)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return False
    else:
        print("Model architecture issue.")
        return False

    model.eval()
    
    # Dummy input
    # standard tile size or something reasonable. 
    # Dynamic axes allow variable resolution.
    dummy_input = torch.randn(1, 3, 64, 64, device='cpu') 
    
    print(f"Exporting to ONNX: {output_path}")
    
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        print("Export successful!")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RealESRGAN to ONNX")
    parser.add_argument('--model', default='models/RealESRGAN_x4plus.pth', help='Path to .pth model')
    parser.add_argument('--output', default='models/realesrgan.onnx', help='Path to output .onnx')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model path does not exist: {args.model}")
        # Try finding it in the default location if the user just cloned it
        # But for now fail.
        sys.exit(1)
        
    export_onnx(args.model, args.output, args.opset)
