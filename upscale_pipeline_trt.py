import cv2
import torch
import tensorrt as trt
import argparse
import sys
import numpy as np
from realesrgan import RealESRGANer
from tqdm import tqdm

class TRTModel(torch.nn.Module):
    def __init__(self, engine_path, device='cuda'):
        super(TRTModel, self).__init__()
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)
        self.device = device
        
        # Get tensor names
        self.input_name = None
        self.output_name = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

    def eval(self):
        pass # always eval

    def load_state_dict(self, state_dict, strict=True):
        pass # Ignore weights loading

    def forward(self, x):
        # x is (B, 3, H, W)
        shape = tuple(x.shape)
        self.context.set_input_shape(self.input_name, shape)
        
        # Output shape calc (4x scale)
        out_h = shape[2] * 4
        out_w = shape[3] * 4
        output = torch.empty((shape[0], 3, out_h, out_w), dtype=x.dtype, device=self.device)
        
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())
        
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        
        return output

def process_video_gen(input_path, output_path, engine_path, tile_size=512, yield_results=True):
    """
    Generator that processes video and yields (progress, frame_image) tuples.
    If output_path is provided, writes to file.
    """
    if not os.path.exists(input_path):
        yield (0.0, None)
        return

    print(f"Loading engine: {engine_path}")
    model = TRTModel(engine_path)
    
    upsampler = RealESRGANer(
        scale=4,
        model_path='models/RealESRGAN_x4plus.pth', 
        model=model, 
        tile=tile_size, 
        tile_pad=10, 
        pre_pad=0, 
        half=False, 
        device=torch.device('cuda')
    )
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_width = width * 4
    out_height = height * 4
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    print(f"Processing {input_path}")
    
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # RealESRGANer returns RGB or BGR? 
                # OpenCV is BGR. RealESRGANer expects BGR by default if loaded with cv2.
                # output_frame is BGR.
                output_frame, _ = upsampler.enhance(frame, outscale=4)
                
                if writer:
                    writer.write(output_frame)
                    
                processed_count += 1
                progress = processed_count / total_frames if total_frames > 0 else 0
                
                if yield_results:
                    # Yield every frame or decimate? Let consumer decide or yield all.
                    # Convert BGR to RGB for UI
                    frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                    yield (progress, frame_rgb)
                    
            except RuntimeError as e:
                print(f"Error: {e}")
                break
            
    finally:
        cap.release()
        if writer:
            writer.release()
            
    yield (1.0, None) # Done

def main():
    parser = argparse.ArgumentParser(description="Upscale video using TensorRT and RealESRGANer Tiling")
    parser.add_argument('input', help='Input video path')
    parser.add_argument('--engine', required=True, help='TensorRT engine path')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--tile', type=int, default=512, help='Tile size (should match engine opt profile)')
    
    args = parser.parse_args()
    
    output_path = args.output if args.output else f"{os.path.splitext(args.input)[0]}_trt_tiled.mp4"
    
    pbar = tqdm(unit='frame')
    for progress, _ in process_video_gen(args.input, output_path, args.engine, args.tile, yield_results=False):
        pbar.update(1) # This simple update is inaccurate because generator yields frame by frame but pbar needs incremental.
        # Check specific progress reporting if needed. 
        # Actually our generator yields per processed frame.
        pass
        
    pbar.close()
    print("Done.")

if __name__ == "__main__":
    import os # Ensure os is imported
    main()
