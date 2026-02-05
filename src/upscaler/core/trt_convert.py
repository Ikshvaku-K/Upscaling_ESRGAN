import tensorrt as trt
import os
import argparse
import sys

def build_engine(onnx_file_path, engine_file_path, fp16=False, verbose=False):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # Explicit batch definition is required for ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    config = builder.create_builder_config()
    
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    if not os.path.exists(onnx_file_path):
        print(f"ONNX file {onnx_file_path} not found.")
        return False
        
    print(f"Parsing ONNX file: {onnx_file_path}")
    # Use parse_from_file to correctly handle external data/weights
    if not parser.parse_from_file(onnx_file_path):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return False
            
    # Optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    # Input name is 'input', shape (Batch, 3, Height, Width)
    # We define min, opt, max shapes.
    # Assuming input name is 'input' based on export script.
    
    # Min: 1 frame, very small tile
    # Min: 16x16 (Handles small edge tiles)
    # Opt: 512x512
    # Max: 640x640 (Handles 512 tile + padding e.g. 532)
    profile.set_shape("input", (1, 3, 16, 16), (1, 3, 512, 512), (1, 3, 640, 640))
    config.add_optimization_profile(profile)

    if fp16:
        print("Enabling FP16 precision.")
        config.set_flag(trt.BuilderFlag.FP16)
        
    # Memory pool limit (24GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 24 * 1024 * 1024 * 1024)
        
    # Memory pool limit (optional, but good for 5090 lol)
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 * 1024 * 1024) 

    print("Building TensorRT engine... this may take a few minutes.")
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build engine.")
        return False
        
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"Engine saved to {engine_file_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument('--onnx', default='models/realesrgan.onnx', help='Path to .onnx model')
    parser.add_argument('--output', default='models/realesrgan.trt', help='Path to output .trt engine')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    build_engine(args.onnx, args.output, args.fp16, args.verbose)
