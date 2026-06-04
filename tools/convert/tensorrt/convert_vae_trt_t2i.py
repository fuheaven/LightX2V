#!/usr/bin/env python3
"""
Convert Qwen Image VAE to TensorRT (T2I Optimized).
Supports 5 fixed resolutions for T2I.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt
from loguru import logger
from diffusers import AutoencoderKLQwenImage

# Patch upsample to avoid 'nearest-exact' which is unsupported in many ONNX opsets
orig_interpolate = F.interpolate
def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode == 'nearest-exact':
        mode = 'nearest'
    return orig_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
F.interpolate = patched_interpolate

# T2I Resolutions
T2I_RESOLUTIONS = [
    # (name, height, width, profile_idx)
    ("T2I_16_9", 928, 1664, 0),
    ("T2I_9_16", 1664, 928, 1),
    ("T2I_1_1", 1328, 1328, 2),
    ("T2I_4_3", 1140, 1472, 3),
    ("T2I_3_4", 1024, 768, 4),
]

class EncoderWrapper(nn.Module):
    def __init__(self, encoder, quant_conv):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
    def forward(self, x):
        return self.quant_conv(self.encoder(x))

class DecoderWrapper(nn.Module):
    def __init__(self, decoder, post_quant_conv):
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv
    def forward(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

def export_onnx(model, onnx_path, is_decoder=False):
    """Export VAE component to ONNX."""
    device = torch.device("cuda")
    model.to(device).eval().half()
    
    if is_decoder:
        # Decoder input: [B, 16, 1, H/16, W/16]
        dummy_input = torch.randn(1, 16, 1, 64, 64, device=device, dtype=torch.float16)
        input_names = ["latents"]
        output_names = ["images"]
        dynamic_axes = {"latents": {0: "batch", 3: "height", 4: "width"}}
    else:
        # Encoder input: [B, 3, 1, H, W]
        dummy_input = torch.randn(1, 3, 1, 512, 512, device=device, dtype=torch.float16)
        input_names = ["images"]
        output_names = ["latents"]
        dynamic_axes = {"images": {0: "batch", 3: "height", 4: "width"}}

    logger.info(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=15, # Use a stable opset
        do_constant_folding=True,
    )
    logger.info(f"Exported ONNX to {onnx_path}")

def build_engine(onnx_path, engine_path, is_decoder=False, workspace_gb=8):
    """Build TensorRT engine with multi-profiles for T2I."""
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    
    # Load ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(f"Parser error: {parser.get_error(error)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Create optimization profiles
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    for name, h, w, idx in T2I_RESOLUTIONS:
        profile = builder.create_optimization_profile()
        if is_decoder:
            lh, lw = h // 8, w // 8
            profile.set_shape(input_name, (1, 16, 1, lh, lw), (1, 16, 1, lh, lw), (1, 16, 1, lh, lw))
        else:
            profile.set_shape(input_name, (1, 3, 1, h, w), (1, 3, 1, h, w), (1, 3, 1, h, w))
        config.add_optimization_profile(profile)
        logger.info(f"Added profile {idx}: {name} ({h}x{w})")

    # Build
    logger.info(f"Building TRT engine: {engine_path} ...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        logger.error("Failed to build serialized network")
        return None
        
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    logger.info(f"Saved TRT engine to {engine_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    vae = AutoencoderKLQwenImage.from_pretrained(args.vae_path).cuda().half()
    
    # Encoder
    enc_wrapper = EncoderWrapper(vae.encoder, vae.quant_conv).eval()
    enc_onnx = os.path.join(args.output_dir, "vae_encoder_t2i.onnx")
    export_onnx(enc_wrapper, enc_onnx, is_decoder=False)
    build_engine(enc_onnx, os.path.join(args.output_dir, "vae_encoder_t2i.trt"), is_decoder=False)
    
    # Decoder
    dec_wrapper = DecoderWrapper(vae.decoder, vae.post_quant_conv).eval()
    dec_onnx = os.path.join(args.output_dir, "vae_decoder_t2i.onnx")
    export_onnx(dec_wrapper, dec_onnx, is_decoder=True)
    build_engine(dec_onnx, os.path.join(args.output_dir, "vae_decoder_t2i.trt"), is_decoder=True)

if __name__ == "__main__":
    main()
