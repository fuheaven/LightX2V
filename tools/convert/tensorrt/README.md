# VAE TensorRT Conversion Tool

Convert Qwen-Image VAE to TensorRT engines for accelerated inference (2x+ speedup).

## Requirements

```bash
pip install tensorrt tensorrt-cu12-bindings tensorrt-cu12-libs
pip install onnx onnxruntime onnxscript
```

## Quick Start

### Multi-Ratio Mode (Recommended)

Build engines for 8 common aspect ratios. Best for variable-resolution inputs like image editing.

```bash
python convert_vae_trt.py \
    --model_path /path/to/Qwen-Image-Edit-2511 \
    --output_dir /path/to/vae_trt_engines \
    --multi_ratio
```

**Build time:** ~25-30 minutes

### Static Resolution Mode

Build engine for a single fixed resolution.

```bash
python convert_vae_trt.py \
    --model_path /path/to/Qwen-Image-Edit-2511 \
    --output_dir /path/to/vae_trt_engines \
    --height 1024 --width 1024
```

**Build time:** ~5 minutes

## Command Line Options

| Option | Description |
|--------|-------------|
| `--model_path` | Path to Qwen-Image model directory (required) |
| `--output_dir` | Output directory for TensorRT engines (required) |
| `--multi_ratio` | Build engines for multiple aspect ratios |
| `--height` | Height for static mode (default: 1024) |
| `--width` | Width for static mode (default: 1024) |
| `--build_decoder` | Also build decoder engines (optional) |
| `--force` | Force rebuild existing engines |

## Output Files

### Multi-Ratio Mode

```
output_dir/
├── vae_encoder_1_1_1024.trt   # 1024x1024 (1:1)
├── vae_encoder_1_1_512.trt    # 512x512 (1:1)
├── vae_encoder_4_3_1024.trt   # 1024x768 (4:3)
├── vae_encoder_3_4_1024.trt   # 768x1024 (3:4)
├── vae_encoder_16_9_1152.trt  # 1152x640 (~16:9)
├── vae_encoder_9_16_1152.trt  # 640x1152 (~9:16)
├── vae_encoder_3_2_1024.trt   # 1024x672 (~3:2)
└── vae_encoder_2_3_1024.trt   # 672x1024 (~2:3)
```

### Static Mode

```
output_dir/
├── vae_encoder.trt
└── vae_decoder.trt  # if --build_decoder
```

## Usage in LightX2V

Add to your config JSON:

```json
{
    "vae_type": "tensorrt",
    "trt_vae_config": {
        "multi_ratio": true,
        "engine_dir": "/path/to/vae_trt_engines"
    }
}
```

## Performance

| Resolution | PyTorch | TensorRT | Speedup |
|------------|---------|----------|---------|
| 1024x1024 | 45 ms | 22 ms | **2.0x** |
| 768x1024 | 35 ms | 17 ms | **2.1x** |
| 1280x720 | 38 ms | 20 ms | **1.9x** |

## Related Documentation

- [VAE TensorRT Optimization Guide (中文)](../../../examples/BeginnerGuide/ZH_CN/QwenImageVAETensorRTOptimize.md)
- [VAE TensorRT Optimization Guide (English)](../../../examples/BeginnerGuide/EN/QwenImageVAETensorRTOptimize.md)
