"""
Qwen Image I2I Disagg request script.
Request flow:
  1. Send request to Transformer FIRST (so it blocks on receive_encoder_outputs)
  2. Send the same request to Encoder (text + image encoding then Mooncake send)
  3. Poll Transformer for task completion (image saved there)
"""
import base64
import time

import requests
from loguru import logger

TRANSFORMER_URL = "http://localhost:8013"
ENCODER_URL = "http://localhost:8012"
ENDPOINT = "/v1/tasks/image/"

# Set to your input image path for I2I (or pass via env/arg)
IMAGE_PATH = "/home/fuhaiwen/LightX2V/scripts/qwen_image/optimize/images/test.png"


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def poll_task(url, task_id, timeout=300, interval=5):
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
    if not IMAGE_PATH:
        raise ValueError("Set IMAGE_PATH to your input image for I2I, or pass image_path in payload.")

    payload = {
        "prompt": "Change the person to a standing position, bending over to hold the dog's front paws.",
        "negative_prompt": "",
        "image_path": image_to_base64(IMAGE_PATH),
        "seed": 42,
        "save_result_path": "qwen_i2i_disagg.png",
        "aspect_ratio": "16:9",
    }

    logger.info("Sending request to Transformer...")
    resp_t = requests.post(f"{TRANSFORMER_URL}{ENDPOINT}", json=payload, timeout=30)
    transformer_task_id = resp_t.json().get("task_id")
    logger.info(f"Transformer task_id: {transformer_task_id}")

    logger.info("Sending request to Encoder...")
    resp_e = requests.post(f"{ENCODER_URL}{ENDPOINT}", json=payload, timeout=30)
    logger.info(f"Encoder response: {resp_e.json()}")

    logger.info("Polling Transformer for completion...")
    t_start = time.time()
    result = poll_task(TRANSFORMER_URL, transformer_task_id)
    elapsed = time.time() - t_start
    logger.info(f"I2I Disagg completed in {elapsed:.2f}s: {result}")
