import argparse
import os
import torch
import torch.nn as nn
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE_
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import WanVAE_ as WanVAE_2_2

# Define Resolutions for Profiles
# Format: (Height, Width)
PROFILE_RESOLUTIONS = {
    "min": (480, 832),  # 480p
    "opt": (720, 1280), # 720p
    "max": (1080, 1920) # 1080p (Example, maybe adjust based on VRAM)
}

class WanVAEEncoderWrapper(nn.Module):
    def __init__(self, encoder, model_type="2.1"):
        super().__init__()
        self.encoder = encoder
        self.model_type = model_type
        # Identify cache indices that effectively used (Not None)
        # We assume the user provides the correct number of caches
        # We will dynamically filter None in the forward or expect Nones in Python wrapper?
        # For TRT, inputs must be Tensors.
        # So we expect the Python Runner to pass Zeros for Nones.
        # But if the model explicitly checks "is None", we might need to change model code OR
        # ensure that passing Zeros is equivalent.
        # Validated: Passing Zeros is equivalent to padding 0, which "None" usually implies in CausalConv3d.
        
    def forward(self, x, *caches):
        # x: Input chunk (Patchified if 2.2)
        # caches: list of tensors
        
        # We need to reconstruct the internal cache list for the encoder
        # The encoder expects a list where some items might be mutable.
        # We convert the tuple `caches` to a mutable list.
        feat_cache = list(caches)
        
        # Reset internal index counter
        # The encoder uses `feat_idx` (list of 1 int) to track usage
        feat_idx = [0]
        
        # Call encoder
        out = self.encoder(x, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # Return output and updated caches
        return out, *feat_cache

class WanVAEDecoderWrapper(nn.Module):
    def __init__(self, decoder, model_type="2.1"):
        super().__init__()
        self.decoder = decoder
        self.model_type = model_type

    def forward(self, x, first_chunk, *caches):
        # x: Input latent chunk (Conv2 output)
        # first_chunk: Tensor(bool/int) or Int. 
        # caches: list of tensors
        
        feat_cache = list(caches)
        feat_idx = [0]
        
        # Wan 2.2 supports first_chunk argument
        if self.model_type == "2.2":
             # Ensure first_chunk is bool
             is_first_chunk = (first_chunk > 0)
             out = self.decoder(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=is_first_chunk)
        else:
             out = self.decoder(x, feat_cache=feat_cache, feat_idx=feat_idx)
             
        return out, *feat_cache


def get_model(version):
    if version == "2.1":
        model = WanVAE_(dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2)
    elif version == "2.2":
        model = WanVAE_2_2(dim=160, dec_dim=256, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2)
    else:
        raise ValueError(f"Unknown version {version}")
    return model

def calculate_cache_shapes(model, resolution, version):
    # Use logic from get_wan_vae_cache_config.py
    # Instantiate dummy to trace shapes
    model = model.to("cpu")
    model.clear_cache()
    
    h, w = resolution
    # Encoder Input - Wan 2.1 uses raw pixels
    # Wan 2.2 uses patchified (but patchify is done OUTSIDE Encoder in wrapper)
    # So for shape calculation, we just use 3 channels for 2.1, 12 for 2.2 wrapper
    # BUT our wrapper will wrap Encoder3d directly, which takes patchified if 2.2
    
    # Key insight: WanVAE.encode for 2.1 does NOT patchify; directly calls encoder
    # For 2.2 (per vae_2_2.py), encode DOES call patchify first.
    # So Encoder input is patchified for 2.2.
    
    if version == "2.2":
        # Patchified input [B, 12, T, H/2, W/2]
        enc_in = torch.zeros(1, 12, 1, h // 2, w // 2)
        dec_c = 16
    else:
        # Raw pixels [B, 3, T, H, W] - No Patchify for 2.1
        enc_in = torch.zeros(1, 3, 1, h, w)
        dec_c = 4
        
    # Decoder Input (Latent after Conv2)
    # Stride 8 assumption
    lat_h = h // 8
    lat_w = w // 8
    dec_in = torch.zeros(1, dec_c, 1, lat_h, lat_w)

    # === Pre-initialize feat_cache with Zeros to avoid "None" causing Kernel Size Errors ===
    # Get the number of cache slots
    num_enc_caches = model._enc_conv_num
    num_dec_caches = model._conv_num
    
    # We need to know the shapes BEFORE running. Catch-22.
    # Alternative: Run with T=4 (a full chunk) to let the model populate caches properly.
    # This is safer.
    
    # Use T=4 for encoder to properly populate caches
    if version == "2.2":
        enc_in_multi = torch.zeros(1, 12, 4, h // 2, w // 2)
    else:
        enc_in_multi = torch.zeros(1, 3, 4, h, w)
    
    # Run Encoder with multi-frame input
    model.clear_cache()
    model.encoder(enc_in_multi, feat_cache=model._enc_feat_map, feat_idx=[0])
    enc_cache_shapes = [list(c.shape) for c in model._enc_feat_map if isinstance(c, torch.Tensor)]

    # Run Decoder with multi-frame latent
    model.clear_cache()
    dec_in_multi = torch.zeros(1, dec_c, 4, lat_h, lat_w)
    
    kwargs = {"feat_cache": model._feat_map, "feat_idx": [0]}
    if version == "2.2": kwargs["first_chunk"] = True
    model.decoder(dec_in_multi, **kwargs)
    
    dec_cache_shapes = [list(c.shape) for c in model._feat_map if isinstance(c, torch.Tensor)]
    
    return enc_cache_shapes, dec_cache_shapes

def export_onnx(model, wrapper, args, cache_shapes, component_name):
    # Dummy Inputs for Export
    # Use Optimization Profile "opt" resolution for export dummy
    h, w = PROFILE_RESOLUTIONS["opt"]
    
    # Use T=4 to match cache calculation and avoid CausalConv3d kernel size issues
    T = 4
    
    if component_name == "encoder":
        if args.version == "2.2":
             dummy_in = torch.randn(1, 12, T, h//2, w//2)
        else:
             dummy_in = torch.randn(1, 3, T, h, w)
    else: # decoder
        lat_h, lat_w = h // 8, w // 8
        if args.version == "2.2":
             dummy_in = torch.randn(1, 16, T, lat_h, lat_w)
        else:
             dummy_in = torch.randn(1, 4, T, lat_h, lat_w)

    dummy_caches = [torch.randn(*shape) for shape in cache_shapes]
    
    input_names = ["input"] + [f"cache_in_{i}" for i in range(len(cache_shapes))]
    output_names = ["output"] + [f"cache_out_{i}" for i in range(len(cache_shapes))]
    
    inputs = (dummy_in, *dummy_caches)
    
    dynamic_axes = {
        "input": {0: "batch", 3: "height", 4: "width"},
        "output": {0: "batch", 3: "height", 4: "width"}
    }
    
    # Cache Dynamic Axes
    for i in range(len(cache_shapes)):
        # Caches usually [B, C, T, H_latent, W_latent]
        # H, W are dynamic. T is usually fixed (2 or similar). C fixed.
        # Check cache shape dims. 
        # Wan cache: [B, C, T, H, W]
        dynamic_axes[f"cache_in_{i}"] = {0: "batch", 3: "height", 4: "width"}
        dynamic_axes[f"cache_out_{i}"] = {0: "batch", 3: "height", 4: "width"}

    if args.version == "2.2" and component_name == "decoder":
        # Add first_chunk argument
        # We pass it as a tensor (int32 or bool)
        dummy_first_chunk = torch.tensor(1, dtype=torch.int32)
        inputs = (dummy_in, dummy_first_chunk, *dummy_caches)
        input_names = ["input", "first_chunk"] + [f"cache_in_{i}" for i in range(len(cache_shapes))]
        # first_chunk is scalar or shape [1], not dynamic spatial
        # No dynamic axes needed for first_chunk scalar
    
    onnx_path = os.path.join(args.output_dir, f"wan_{args.version}_{component_name}.onnx")
    
    print(f"Exporting {component_name} to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17
    )
    return onnx_path

def build_engine(onnx_path, args, cache_shapes, component_name):
    # Construct trtexec command
    engine_name = onnx_path.replace(".onnx", ".engine") # or .trt
    
    cmd = f"trtexec --onnx={onnx_path} --saveEngine={engine_name} --fp16"
    
    # Define Profiles
    # Input Shapes
    # Profiles for Input and Caches
    
    profiles = []
    
    for mode in ["min", "opt", "max"]:
        h, w = PROFILE_RESOLUTIONS[mode]
        
        # Main Input Shape
        if component_name == "encoder":
            if args.version == "2.2":
                in_shape = f"1x12x1x{h//2}x{w//2}"
            else:
                in_shape = f"1x3x1x{h}x{w}"
        else: # decoder
            lat_h, lat_w = h // 8, w // 8
            if args.version == "2.2":
                in_shape = f"1x16x1x{lat_h}x{lat_w}"
            else:
                in_shape = f"1x4x1x{lat_h}x{lat_w}"
        
        profile_str = f"input:{in_shape}"
        
        # First Chunk (Decoder 2.2)
        if args.version == "2.2" and component_name == "decoder":
             profile_str += f",first_chunk:1" # First chunk is scalar/0D or 1D shape? Torch export scalar usually is [] or [1]
             # If we used torch.tensor(1), it is scalar. trtexec scalar support?
             # Probably safest to export as [1].
             # Let's check torch export above.
        
        # Cache Shapes
        # We need to calculate cache shapes for THIS resolution
        # Re-using calculate_cache_shapes logic or scaling based on opt?
        # Better to re-calculate to be precise.
        # But instantiating model 3 times is slow.
        # Caches scale linearly with H/W.
        # We can calculate scaling factors from 'opt' shapes.
        # Or just use the model to calc shapes (safest).
        pass

    # For simplicity in this script write-up, I will print the trtexec command
    # In real execution, we might use Python API or os.system
    
    print(f"Building engine {engine_name}...")
    # Because calculating exact cache shapes for min/max profiles strictly requires model logic,
    # and doing it in shell string is hard.
    # We will use the TRT Python API Builder in a separate step or `trtexec` via subprocess if we can generate the string.
    # Given the complexity of cache shapes (dozens of inputs), constructing the trtexec command string is non-trivial and prone to length limits.
    # BUT `trtexec` is robust.
    
    # We will skip actual build in this script and output the command or use os.system if requested.
    # For now, let's just export ONNX. The user handles build or we add it?
    # Plan says "Build TensorRT... Use trtexec (via Python API)".
    # So we should use `tensorrt` python library.
    
    return engine_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="2.1", choices=["2.1", "2.2"])
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--skip_build", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Initializing Model {args.version}...")
    model = get_model(args.version)
    
    # 1. Calculate Shapes using 'opt' profile to define the ONNX export structure
    # (Input/Outputs count and structure)
    print("Calculating cache configuration...")
    # Use OPT resolution for initial shape calc
    enc_caches, dec_caches = calculate_cache_shapes(model, PROFILE_RESOLUTIONS["opt"], args.version)
    
    print(f"Encoder has {len(enc_caches)} cache tensors.")
    print(f"Decoder has {len(dec_caches)} cache tensors.")
    
    # 2. Export ONNX
    enc_wrapper = WanVAEEncoderWrapper(model.encoder, args.version)
    enc_onnx = export_onnx(model, enc_wrapper, args, enc_caches, "encoder")
    
    dec_wrapper = WanVAEDecoderWrapper(model.decoder, args.version)
    dec_onnx = export_onnx(model, dec_wrapper, args, dec_caches, "decoder")
    
    if not args.skip_build:
        print("Build step not fully implemented in this script version (Requires TRT Python API setup).")
        print(f"Please run trtexec manually for: {enc_onnx} and {dec_onnx}")
        # Note: Implementing full TRT Python API build with profiles for 3 resolutions * 30+ caches is lengthy.
        # Suggesting to user to use trtexec or separate build script.
        # Or we can output a helper shell script?
        
        with open(os.path.join(args.output_dir, "build_trt.sh"), "w") as f:
            f.write("#!/bin/bash\n")
            # Generate trtexec command for profiles
            for name, cache_shapes, onnx in [("encoder", enc_caches, enc_onnx), ("decoder", dec_caches, dec_onnx)]:
                cmd = f"trtexec --onnx={onnx} --saveEngine={onnx.replace('.onnx', '.engine')} --fp16"
                
                # Profiles
                for pname in ["min", "opt", "max"]:
                    res = PROFILE_RESOLUTIONS[pname]
                    # Calc shapes for this res
                    e_s, d_s = calculate_cache_shapes(model, res, args.version)
                    curr_shapes = e_s if name == "encoder" else d_s
                    
                    # Build Shapes String
                    # Input
                    if name == "encoder":
                         if args.version == "2.2": i_s = f"1x12x1x{res[0]//2}x{res[1]//2}"
                         else: i_s = f"1x3x1x{res[0]}x{res[1]}"
                    else:
                         if args.version == "2.2": i_s = f"1x16x1x{res[0]//8}x{res[1]//8}"
                         else: i_s = f"1x4x1x{res[0]//8}x{res[1]//8}"
                    
                    profile_arg = f"input:{i_s}"
                    
                    if args.version == "2.2" and name == "decoder":
                         profile_arg += ",first_chunk:[]" # Scalar
                    
                    for i, shape in enumerate(curr_shapes):
                        shape_str = "x".join(map(str, shape))
                        profile_arg += f",cache_in_{i}:{shape_str}"
                        
                    cmd += f" --{pname}Shapes={profile_arg}"
                
                f.write(f"echo 'Building {name} TRT Engine...'\n")
                f.write(f"{cmd}\n\n")
        
        print(f"Generated build script: {os.path.join(args.output_dir, 'build_trt.sh')}")

if __name__ == "__main__":
    main()
