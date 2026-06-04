#!/usr/bin/env python3
"""
VAE TensorRT Engine Conversion Tool

Convert Qwen-Image VAE to TensorRT engines for accelerated inference.
Supports both static shape and multi-profile modes.

Usage:
    # Multi-profile mode (recommended)
    python convert_vae_trt.py --model_path /path/to/model --output_dir /path/to/output --multi_profile --build_decoder

    # Static shape mode
    python convert_vae_trt.py --model_path /path/to/model --output_dir /path/to/output --height 1024 --width 1024
"""

import argparse
import os
import torch
from loguru import logger

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    logger.error("TensorRT not available. Please install: pip install tensorrt tensorrt-cu12-bindings tensorrt-cu12-libs")


# Profiles used in the multi-profile engine (must match build script)
# Max supported: 1920x1920 (due to TRT build memory constraints)
PROFILE_CONFIGS = [
    # (name, height, width)
    ("1_1_512", 512, 512),
    ("1_1_1024", 1024, 1024),
    ("16_9_480p", 480, 848),
    ("16_9_720p", 720, 1280),
    ("16_9_1080p", 1080, 1920),     # Max width 1920
    ("9_16_720p", 1280, 720),
    ("9_16_1080p", 1920, 1080),     # Max height 1920
    ("4_3_768p", 768, 1024),
    ("3_2_1080p", 1080, 1620),
]


def get_trt_logger():
    return trt.Logger(trt.Logger.WARNING)


def export_vae_to_onnx(vae, output_path, height, width, component="encoder"):
    """Export VAE component to ONNX using dynamo exporter.
    
    For encoder: Creates a wrapper that includes both encoder and quant_conv
    to match the full encode() behavior.
    """
    import torch.nn as nn
    
    if component == "encoder":
        # Create wrapper module that includes both encoder and quant_conv
        class EncoderWrapper(nn.Module):
            def __init__(self, encoder, quant_conv):
                super().__init__()
                self.encoder = encoder
                self.quant_conv = quant_conv
                
            def forward(self, x):
                # Simplified single-frame encoding (no temporal caching)
                # Note: This assumes x has shape [B, C, 1, H, W] (single frame)
                out = self.encoder(x)
                enc = self.quant_conv(out)
                return enc
        
        model = EncoderWrapper(vae.encoder, vae.quant_conv)
        model.eval()
        dummy_input = torch.randn(1, 3, 1, height, width, device="cuda", dtype=torch.float16)
    else:
        # Create wrapper module for Decoder too (must include post_quant_conv)
        class DecoderWrapper(nn.Module):
            def __init__(self, decoder, post_quant_conv):
                super().__init__()
                self.decoder = decoder
                self.post_quant_conv = post_quant_conv
            
            def forward(self, z):
                z = self.post_quant_conv(z)
                dec = self.decoder(z)
                return dec

        model = DecoderWrapper(vae.decoder, vae.post_quant_conv)
        model.eval()
        
        # Decoder input shape: latent space
        latent_h, latent_w = height // 8, width // 8
        dummy_input = torch.randn(1, 16, 1, latent_h, latent_w, device="cuda", dtype=torch.float16)

    logger.info(f"Exporting {component} to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
    )
    logger.info(f"ONNX export complete: {output_path}")


def build_trt_engine(onnx_path, engine_path, workspace_gb=4):
    """Build TensorRT engine from ONNX model."""
    trt_logger = get_trt_logger()
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    logger.info(f"Parsing ONNX: {onnx_path}")
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            logger.error(f"Parse error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")

    logger.info("Building TensorRT engine (this may take several minutes)...")
    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    logger.info(f"Engine saved: {engine_path}")
    return engine_path


def convert_single_resolution(model_path, output_dir, height, width, build_decoder=False):
    """Convert VAE for a single resolution."""
    from diffusers import AutoencoderKLQwenImage

    os.makedirs(output_dir, exist_ok=True)

    # Load VAE
    vae_path = os.path.join(model_path, "vae")
    if not os.path.exists(vae_path):
        vae_path = model_path
    
    logger.info(f"Loading VAE from: {vae_path}")
    vae = AutoencoderKLQwenImage.from_pretrained(vae_path).to("cuda").to(torch.float16)
    vae.eval()

    # Export and build encoder
    encoder_onnx = os.path.join(output_dir, "vae_encoder.onnx")
    encoder_trt = os.path.join(output_dir, "vae_encoder.trt")
    
    export_vae_to_onnx(vae, encoder_onnx, height, width, "encoder")
    build_trt_engine(encoder_onnx, encoder_trt)

    # Optionally build decoder
    if build_decoder:
        decoder_onnx = os.path.join(output_dir, "vae_decoder.onnx")
        decoder_trt = os.path.join(output_dir, "vae_decoder.trt")
        
        export_vae_to_onnx(vae, decoder_onnx, height, width, "decoder")
        build_trt_engine(decoder_onnx, decoder_trt)

    # Cleanup
    del vae
    torch.cuda.empty_cache()

    logger.info(f"Conversion complete. Output: {output_dir}")


# ============================================================================
# Dynamic Shape Mode (Experimental)
# ============================================================================

def export_vae_to_onnx_dynamic(vae, output_path, component="encoder"):
    """Export VAE component to ONNX with dynamic spatial dimensions.
    
    Uses torch.onnx.export with dynamo=True and dynamic_shapes parameter.
    """
    import torch.nn as nn
    
    if component == "encoder":
        class EncoderWrapper(nn.Module):
            def __init__(self, encoder, quant_conv):
                super().__init__()
                self.encoder = encoder
                self.quant_conv = quant_conv
            def forward(self, x):
                return self.quant_conv(self.encoder(x))
        
        model = EncoderWrapper(vae.encoder, vae.quant_conv).eval()
        # Encoder input: [B, C, F, H, W] where H,W are dynamic
        dummy_input = torch.randn(1, 3, 1, 1024, 1024, device="cuda", dtype=torch.float16)
    else:
        class DecoderWrapper(nn.Module):
            def __init__(self, decoder, post_quant_conv):
                super().__init__()
                self.decoder = decoder
                self.post_quant_conv = post_quant_conv
            def forward(self, z):
                z = self.post_quant_conv(z)
                return self.decoder(z)

        model = DecoderWrapper(vae.decoder, vae.post_quant_conv).eval()
        # Decoder input: [B, C, F, H/8, W/8] latent space
        dummy_input = torch.randn(1, 16, 1, 128, 128, device="cuda", dtype=torch.float16)
    
    logger.info(f"Exporting {component} to ONNX with dynamic shapes: {output_path}")
    
    # Use dynamic_axes for dynamo export (dynamic_shapes requires torch.export.Dim which is complex)
    # Fall back to standard dynamic_axes approach
    dynamic_axes = {
        "input": {3: "height", 4: "width"},
        "output": {3: "out_height", 4: "out_width"}
    }
    
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=True,
    )
    logger.info(f"Dynamic ONNX export complete: {output_path}")


def build_trt_engine_dynamic(onnx_path, engine_path, component="encoder", workspace_gb=8):
    """Build TensorRT engine with optimization profile for dynamic shapes.
    
    Creates a single engine that supports multiple input resolutions.
    """
    trt_logger = get_trt_logger()
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    
    logger.info(f"Parsing ONNX: {onnx_path}")
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            logger.error(f"Parse error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")
    
    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    
    # Get input tensor name
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    logger.info(f"Input tensor: {input_name}, shape: {input_tensor.shape}")
    
    if component == "encoder":
        # Encoder input: [1, 3, 1, H, W] pixel-level
        # Support 512x512 to 2048x2048 (must be divisible by 8)
        min_shape = (1, 3, 1, 512, 512)
        opt_shape = (1, 3, 1, 1024, 1024)
        max_shape = (1, 3, 1, 2048, 2048)
    else:
        # Decoder input: [1, 16, 1, H/8, W/8] latent-level
        min_shape = (1, 16, 1, 64, 64)    # 512x512 pixels
        opt_shape = (1, 16, 1, 128, 128)  # 1024x1024 pixels
        max_shape = (1, 16, 1, 256, 256)  # 2048x2048 pixels
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    logger.info(f"Optimization profile: min={min_shape}, opt={opt_shape}, max={max_shape}")
    logger.info("Building TensorRT engine with dynamic shapes (this may take longer)...")
    
    serialized = builder.build_serialized_network(network, config)
    
    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    with open(engine_path, "wb") as f:
        f.write(serialized)
    
    logger.info(f"Dynamic engine saved: {engine_path}")
    return engine_path


def build_trt_engine_multi_profile(onnx_path, engine_path, component="encoder", workspace_gb=16):
    """Build TensorRT engine with multiple optimization profiles.
    
    Creates a single engine with one profile per common resolution, using a global
    min/max range (64x64 to 12K) to support any resolution within bounds.
    Each profile has a different opt_shape for optimal performance at that resolution.
    """
    trt_logger = get_trt_logger()
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    
    logger.info(f"Parsing ONNX: {onnx_path}")
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            logger.error(f"Parse error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")
    
    # Get input tensor name
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    logger.info(f"Input tensor: {input_name}, shape: {input_tensor.shape}")
    
    # Global min/max range for all profiles
    # Encoder: 64x64 to 1920x1920 pixels (1080p max to avoid GPU OOM during build)
    # Decoder: 8x8 to 240x240 latent (corresponding to 64x64 to 1920x1920 pixels)
    if component == "encoder":
        global_min = (1, 3, 1, 64, 64)
        global_max = (1, 3, 1, 1920, 1920)  # 1080p max
    else:
        global_min = (1, 16, 1, 8, 8)         # 64x64 pixels
        global_max = (1, 16, 1, 240, 240)     # 1920x1920 pixels
    
    logger.info(f"Global range: min={global_min}, max={global_max}")
    
    # Create one optimization profile per common resolution
    for idx, (name, h, w) in enumerate(PROFILE_CONFIGS):
        profile = builder.create_optimization_profile()
        
        if component == "encoder":
            opt_shape = (1, 3, 1, h, w)
        else:
            lat_h, lat_w = h // 8, w // 8
            opt_shape = (1, 16, 1, lat_h, lat_w)
        
        profile.set_shape(input_name, global_min, opt_shape, global_max)
        config.add_optimization_profile(profile)
        
        logger.info(f"  Profile {idx} ({name}): opt={opt_shape}")
    
    logger.info(f"Building TensorRT engine with {len(PROFILE_CONFIGS)} optimization profiles...")
    logger.info("This may take significantly longer than single-profile builds...")
    
    serialized = builder.build_serialized_network(network, config)
    
    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    with open(engine_path, "wb") as f:
        f.write(serialized)
    
    logger.info(f"Multi-profile engine saved: {engine_path}")
    return engine_path


def convert_multi_profile(model_path, output_dir, build_decoder=False):
    """Convert VAE to multi-profile TensorRT engine.
    
    Creates a single engine file with multiple optimization profiles,
    one for each aspect ratio configuration.
    """
    from diffusers import AutoencoderKLQwenImage
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAE
    vae_path = os.path.join(model_path, "vae")
    if not os.path.exists(vae_path):
        vae_path = model_path
    
    logger.info(f"Loading VAE from: {vae_path}")
    vae = AutoencoderKLQwenImage.from_pretrained(vae_path).to("cuda").to(torch.float16)
    vae.eval()
    
    # Export dynamic ONNX first (reuse existing function)
    encoder_onnx = os.path.join(output_dir, "vae_encoder_dynamic.onnx")
    encoder_trt = os.path.join(output_dir, "vae_encoder_multi_profile.trt")
    
    logger.info("=" * 60)
    logger.info("Building Multi-Profile VAE Encoder")
    logger.info("=" * 60)
    
    # Export ONNX if not exists
    if not os.path.exists(encoder_onnx):
        export_vae_to_onnx_dynamic(vae, encoder_onnx, "encoder")
    else:
        logger.info(f"Using existing ONNX: {encoder_onnx}")
    
    build_trt_engine_multi_profile(encoder_onnx, encoder_trt, "encoder") if not os.path.exists(encoder_trt) else logger.info("Encoder TRT exists, skipping build.")
    
    # Optionally build multi-profile decoder
    if build_decoder:
        decoder_onnx = os.path.join(output_dir, "vae_decoder_dynamic.onnx")
        decoder_trt = os.path.join(output_dir, "vae_decoder_multi_profile.trt")
        
        logger.info("=" * 60)
        logger.info("Building Multi-Profile VAE Decoder")
        logger.info("=" * 60)
        
        if not os.path.exists(decoder_onnx):
            export_vae_to_onnx_dynamic(vae, decoder_onnx, "decoder")
        else:
            logger.info(f"Using existing ONNX: {decoder_onnx}")
        
        build_trt_engine_multi_profile(decoder_onnx, decoder_trt, "decoder")
    
    # Cleanup
    del vae
    torch.cuda.empty_cache()
    
    logger.info("=" * 60)
    logger.info("Multi-profile conversion complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Profiles: {len(PROFILE_CONFIGS)} aspect ratios")
    logger.info("=" * 60)


def convert_dynamic(model_path, output_dir, build_decoder=False):
    """Convert VAE to single dynamic-shape TensorRT engine."""
    from diffusers import AutoencoderKLQwenImage
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAE
    vae_path = os.path.join(model_path, "vae")
    if not os.path.exists(vae_path):
        vae_path = model_path
    
    logger.info(f"Loading VAE from: {vae_path}")
    vae = AutoencoderKLQwenImage.from_pretrained(vae_path).to("cuda").to(torch.float16)
    vae.eval()
    
    # Export and build dynamic encoder
    encoder_onnx = os.path.join(output_dir, "vae_encoder_dynamic.onnx")
    encoder_trt = os.path.join(output_dir, "vae_encoder_dynamic.trt")
    
    logger.info("=" * 60)
    logger.info("Building Dynamic VAE Encoder")
    logger.info("=" * 60)
    export_vae_to_onnx_dynamic(vae, encoder_onnx, "encoder")
    build_trt_engine_dynamic(encoder_onnx, encoder_trt, "encoder")
    
    # Optionally build dynamic decoder
    if build_decoder:
        decoder_onnx = os.path.join(output_dir, "vae_decoder_dynamic.onnx")
        decoder_trt = os.path.join(output_dir, "vae_decoder_dynamic.trt")
        
        logger.info("=" * 60)
        logger.info("Building Dynamic VAE Decoder")
        logger.info("=" * 60)
        export_vae_to_onnx_dynamic(vae, decoder_onnx, "decoder")
        build_trt_engine_dynamic(decoder_onnx, decoder_trt, "decoder")
    
    # Cleanup
    del vae
    torch.cuda.empty_cache()
    
    logger.info("=" * 60)
    logger.info("Dynamic conversion complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen-Image VAE to TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-profile mode (recommended for production)
  python convert_vae_trt.py --model_path /path/to/Qwen-Image-Edit-2511 \\
      --output_dir /path/to/vae_trt_engines --multi_profile --build_decoder

  # Static single resolution mode
  python convert_vae_trt.py --model_path /path/to/Qwen-Image-Edit-2511 \\
      --output_dir /path/to/vae_trt_engines --height 1024 --width 1024
        """
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Qwen-Image model directory (containing 'vae' subdirectory)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for TensorRT engines")
    parser.add_argument("--multi_profile", action="store_true",
                        help="Build single engine with multiple optimization profiles (best of both worlds)")
    parser.add_argument("--dynamic", action="store_true",
                        help="Build single dynamic-shape engine with one profile (experimental)")
    parser.add_argument("--height", type=int, default=1024,
                        help="Height for static mode (default: 1024)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Width for static mode (default: 1024)")
    parser.add_argument("--build_decoder", action="store_true",
                        help="Also build decoder engines (optional, for decode acceleration)")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild existing engines")

    # Deprecated args
    parser.add_argument("--multi_ratio", action="store_true",
                        help="Deprecated. Use --multi_profile instead.")

    args = parser.parse_args()

    if not HAS_TRT:
        logger.error("TensorRT is required. Please install it first.")
        return 1

    if args.multi_profile or args.multi_ratio:
        if args.multi_ratio:
           logger.warning("--multi_ratio is deprecated and will be removed. Please use --multi_profile.")
        convert_multi_profile(
            args.model_path,
            args.output_dir,
            build_decoder=args.build_decoder
        )
    elif args.dynamic:
        convert_dynamic(
            args.model_path,
            args.output_dir,
            build_decoder=args.build_decoder
        )
    else:
        convert_single_resolution(
            args.model_path,
            args.output_dir,
            args.height,
            args.width,
            build_decoder=args.build_decoder
        )


if __name__ == "__main__":
    main()
