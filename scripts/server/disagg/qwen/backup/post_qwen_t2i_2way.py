"""
Qwen Image T2I Disagg request script.
Request flow:
  1. Send request to Transformer FIRST (so it blocks on receive_encoder_outputs)
  2. Send the same request to Encoder (text encoding then Mooncake send)
  3. Poll Transformer for task completion (image saved there)
"""
import time

import requests
from loguru import logger

TRANSFORMER_URL = "http://localhost:8003"
ENCODER_URL = "http://localhost:8002"
ENDPOINT = "/v1/tasks/image/"

PAYLOAD = {
    "prompt": "A cute cat sitting on a wooden table, soft lighting, 4k photo.",
    "negative_prompt": "blurry, low quality, distorted",
    "seed": 42,
    "save_result_path": "qwen_t2i_disagg.png",
    "aspect_ratio": "16:9",
}


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
    logger.info("Sending request to Transformer...")
    resp_t = requests.post(f"{TRANSFORMER_URL}{ENDPOINT}", json=PAYLOAD, timeout=30)
    transformer_task_id = resp_t.json().get("task_id")
    logger.info(f"Transformer task_id: {transformer_task_id}")

    logger.info("Sending request to Encoder...")
    resp_e = requests.post(f"{ENCODER_URL}{ENDPOINT}", json=PAYLOAD, timeout=30)
    logger.info(f"Encoder response: {resp_e.json()}")

    logger.info("Polling Transformer for completion...")
    t_start = time.time()
    result = poll_task(TRANSFORMER_URL, transformer_task_id)
    elapsed = time.time() - t_start
    logger.info(f"T2I Disagg completed in {elapsed:.2f}s: {result}")
