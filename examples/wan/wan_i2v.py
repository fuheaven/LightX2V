"""
Wan2.2 image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.2 model for I2V generation.
"""
# import os
# os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"

import torch
try:
    import torch_gcu
    from torch_gcu import transfer_to_gcu
except Exception as e:
    print(f"Warning: {e}")


import sys
sys.path.append("/home/weixing.zhang/test_LightX2V/LightX2V")
from lightx2v import LightX2VPipeline

# Initialize pipeline for Wan2.2 I2V task
# For wan2.1, use model_cls="wan2.1"
pipe = LightX2VPipeline(
    model_path="/home/weixing.zhang/wan2.2/Wan2.2-I2V-A14B",
    model_cls="wan2.2_moe",
    task="i2v",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/wan22/wan_moe_i2v.json"
# )

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",  # For Wan models, supports both "block" and "phase"
    text_encoder_offload=True,
    image_encoder_offload=True,
    vae_offload=True,
)

# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="flash_attn2",
    infer_steps=40,
    height=480,  # Can be set to 720 for higher resolution
    width=832,  # Can be set to 1280 for higher resolution
    num_frames=81,
    guidance_scale=[3.5, 3.5],  # For wan2.1, guidance_scale is a scalar (e.g., 5.0)
    sample_shift=5.0,
)


pipe.enable_parallel(
    seq_p_size=8,                    # 序列并行大小
    seq_p_attn_type="ulysses",       # 序列并行注意力类型
)


# Generation parameters
seed = 42
prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
image_path = "/home/weixing.zhang/test_LightX2V/LightX2V/assets/inputs/imgs/img_0.jpg"
save_result_path = "./output.mp4"




# Generate video
pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
