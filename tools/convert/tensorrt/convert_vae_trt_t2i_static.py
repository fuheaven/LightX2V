#!/usr/bin/env python3
"""
Convert Qwen Image VAE to Static Shape TensorRT engines.
Builds separate engines for each of the 5 T2I resolutions.
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
    ("16_9", 928, 1664),
    ("9_16", 1664, 928),
    ("1_1", 1328, 1328),
    ("4_3", 1140, 1472),
    ("3_4", 1024, 768),
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

def export_onnx(model, onnx_path, h, w, is_decoder=False):
    """Export VAE component to ONNX with static shape."""
    device = torch.device("cuda")
    model.to(device).eval().half()
    
    if is_decoder:
        # Decoder input: [1, 16, 1, H/8, W/8]
        lh, lw = h // 8, w // 8
        dummy_input = torch.randn(1, 16, 1, lh, lw, device=device, dtype=torch.float16)
        input_names = ["latents"]
        output_names = ["images"]
    else:
        # Encoder input: [1, 3, 1, H, W]
        dummy_input = torch.randn(1, 3, 1, h, w, device=device, dtype=torch.float16)
        input_names = ["images"]
        output_names = ["latents"]

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=15,
        do_constant_folding=True,
    )
    logger.info(f"Exported static ONNX to {onnx_path} ({h}x{w})")

def build_engine_static(onnx_path, engine_path, workspace_gb=8):
    """Build static TensorRT engine."""
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(f"Parser error: {parser.get_error(error)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    
    logger.info(f"Building static TRT engine: {engine_path} ...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        return None
        
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    logger.info(f"Saved static TRT engine to {engine_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    vae = AutoencoderKLQwenImage.from_pretrained(args.vae_path).cuda().half()
    enc_wrapper = EncoderWrapper(vae.encoder, vae.quant_conv).eval()
    dec_wrapper = DecoderWrapper(vae.decoder, vae.post_quant_conv).eval()

    for name, h, w in T2I_RESOLUTIONS:
        res_dir = os.path.join(args.output_dir, name)
        os.makedirs(res_dir, exist_ok=True)
        
        # Encoder
        enc_onnx = os.path.join(res_dir, "vae_encoder.onnx")
        enc_engine = os.path.join(res_dir, "vae_encoder.trt")
        export_onnx(enc_wrapper, enc_onnx, h, w, is_decoder=False)
        build_engine_static(enc_onnx, enc_engine)
        
        # Decoder
        dec_onnx = os.path.join(res_dir, "vae_decoder.onnx")
        dec_engine = os.path.join(res_dir, "vae_decoder.trt")
        export_onnx(dec_wrapper, dec_onnx, h, w, is_decoder=True)
        build_engine_static(dec_onnx, dec_engine)

if __name__ == "__main__":
    main()
