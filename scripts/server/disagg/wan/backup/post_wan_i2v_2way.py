"""
Wan2.1 I2V Disagg request script.
The correct disagg request flow:
  1. Send request to Transformer FIRST (so it starts blocking on receive_encoder_outputs)
  2. Send the same request to Encoder (which runs T5/CLIP/VAE encoding then sends via Mooncake)
  3. Poll Transformer for task completion (where the final video is saved)
"""
import base64
import time

import requests
from loguru import logger

TRANSFORMER_URL = "http://localhost:8005"
ENCODER_URL = "http://localhost:8004"
ENDPOINT = "/v1/tasks/video/"

IMAGE_PATH = ""


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def poll_task(url, task_id, timeout=600, interval=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{url}/v1/tasks/{task_id}/status", timeout=10)
        data = r.json()
        status = data.get("status")
        logger.info(f"Task {task_id} status: {status}")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"Task failed: {data.get('error')}")
        time.sleep(interval)
    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")


if __name__ == "__main__":
    payload = {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds.",
        "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "image_path": image_to_base64(IMAGE_PATH),
        "seed": 42,
        "save_result_path": "save_results/wan_i2v_disagg.mp4",
    }

    # Step 1: Send to Transformer first so it starts waiting for Mooncake data
    logger.info("Sending request to Transformer...")
    resp_t = requests.post(f"{TRANSFORMER_URL}{ENDPOINT}", json=payload, timeout=30)
    transformer_task_id = resp_t.json().get("task_id")
    logger.info(f"Transformer task_id: {transformer_task_id}")

    # Step 2: Send same request to Encoder to trigger encoding + Mooncake send
    logger.info("Sending request to Encoder...")
    resp_e = requests.post(f"{ENCODER_URL}{ENDPOINT}", json=payload, timeout=30)
    logger.info(f"Encoder response: {resp_e.json()}")

    # Step 3: Poll Transformer for final completion (video is saved here)
    logger.info("Polling Transformer for completion...")
    t_start = time.time()
    result = poll_task(TRANSFORMER_URL, transformer_task_id)
    elapsed = time.time() - t_start
    logger.info(f"I2V Disagg completed in {elapsed:.2f}s: {result}")
