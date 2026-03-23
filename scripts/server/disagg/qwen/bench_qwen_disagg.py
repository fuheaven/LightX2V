"""
Concurrent benchmark for Qwen Image **disaggregated** deployment (3-way).

Official HTTP flow (examples/BeginnerGuide/.../DisaggSplitDeploy.md):
  1) POST same payload to Decoder -> decoder task_id
  2) POST same payload to Transformer
  3) POST same payload to Encoder
  4) Poll Decoder task_id until completed

Default: full three POSTs per request so Encoder/Transformer receive tasks.
Use --decoder_only only for debugging (standard pipeline will not finish).

Ports:
  - classic multi-replica mode: encoder 8002+i*step, transformer 8003+i*step, decoder 8008+i*step
  - shared-ED multi-transformer mode: one encoder/decoder + multiple transformers (use --trans_url_list)

Examples:
  python bench_qwen_disagg.py --task t2i --replicas 2
  python bench_qwen_disagg.py --task t2i --url_list http://127.0.0.1:8008,http://127.0.0.1:8018
  python bench_qwen_disagg.py --task i2i --url http://127.0.0.1:8014 --image_path path/to/img.jpg
"""

import argparse
import base64
import os
import threading
import time
import uuid
from urllib.parse import urlparse
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Optional

import requests
from loguru import logger


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def process_image_path(image_path: str) -> Any | str:
    if not image_path:
        return image_path
    if image_path.startswith(("http://", "https://")):
        return image_path
    if os.path.exists(image_path):
        return image_to_base64(image_path)
    raise FileNotFoundError(f"Image path not found: {image_path}")


def poll_task(base_url: str, task_id: str, timeout_s: int, interval_s: float) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(f"{base_url}/v1/tasks/{task_id}/status", timeout=10)
        data = r.json()
        status = data.get("status")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"Task failed: {data.get('error')}")
        time.sleep(interval_s)
    raise TimeoutError(f"Task {task_id} timed out after {timeout_s}s")


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    if len(xs_sorted) == 1:
        return xs_sorted[0]
    k = (len(xs_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return d0 + d1


@dataclass
class TaskResult:
    ok: bool
    submit_time: float
    done_time: float
    task_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def latency_s(self) -> float:
        return self.done_time - self.submit_time


def parse_concurrency_list(s: str) -> list[int]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not parts:
        return []
    out: list[int] = []
    for p in parts:
        v = int(p)
        if v <= 0:
            raise ValueError(f"Invalid concurrency value: {v}")
        out.append(v)
    return out


def parse_url_list(s: str) -> list[str]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    return parts


def build_decoder_urls_from_replicas(
    host: str,
    dec_port_base: int,
    port_step: int,
    num_replicas: int,
) -> list[str]:
    """Match start_multi_server_disagg.sh: dec_port = DEC_PORT_BASE + r * PORT_STEP."""
    return [f"http://{host}:{dec_port_base + i * port_step}" for i in range(num_replicas)]


def build_disagg_triples_from_replicas(
    host: str,
    enc_port_base: int,
    trans_port_base: int,
    dec_port_base: int,
    port_step: int,
    num_replicas: int,
) -> list[tuple[str, str, str]]:
    """(encoder, transformer, decoder) base URLs per replica, same layout as start_multi_server_disagg.sh."""
    triples: list[tuple[str, str, str]] = []
    for i in range(num_replicas):
        enc = f"http://{host}:{enc_port_base + i * port_step}"
        trans = f"http://{host}:{trans_port_base + i * port_step}"
        dec = f"http://{host}:{dec_port_base + i * port_step}"
        triples.append((enc, trans, dec))
    return triples


def build_disagg_triples_from_decoder_urls(
    decoder_urls: list[str],
    enc_port_base: int,
    trans_port_base: int,
    port_step: int,
) -> list[tuple[str, str, str]]:
    """Derive encoder/transformer URLs from decoder URL host + port bases (replica index = list index)."""
    triples: list[tuple[str, str, str]] = []
    for i, du in enumerate(decoder_urls):
        p = urlparse(du)
        if not p.scheme or not p.hostname:
            raise ValueError(f"Invalid decoder URL (need scheme and host): {du}")
        if p.port is None:
            raise ValueError(f"Decoder URL must include explicit port: {du}")
        scheme = p.scheme
        host = p.hostname
        dec = f"{scheme}://{host}:{p.port}"
        enc = f"{scheme}://{host}:{enc_port_base + i * port_step}"
        trans = f"{scheme}://{host}:{trans_port_base + i * port_step}"
        triples.append((enc, trans, dec))
    return triples


def build_shared_ed_groups(
    *,
    dec_url: str,
    enc_port_base: int,
    transformer_urls: list[str],
) -> list[tuple[str, str, str]]:
    """Build (enc, trans, dec) groups for one shared ED + multiple transformer URLs."""
    p = urlparse(dec_url)
    if not p.scheme or not p.hostname:
        raise ValueError(f"Invalid decoder URL (need scheme and host): {dec_url}")
    scheme = p.scheme
    host = p.hostname
    enc = f"{scheme}://{host}:{enc_port_base}"
    dec = f"{scheme}://{host}:{p.port}" if p.port is not None else dec_url
    return [(enc, t, dec) for t in transformer_urls]


def post_create_task(base_url: str, endpoint: str, payload: dict) -> str:
    resp = requests.post(f"{base_url}{endpoint}", json=payload, timeout=30)
    resp.raise_for_status()
    task_id = resp.json().get("task_id")
    if not task_id:
        raise RuntimeError(f"No task_id in response from {base_url}: {resp.text}")
    return task_id


def make_task_payload(
    *,
    task_mode: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    save_result_path: str,
    image_b64: Any | str | None,
    aspect_ratio: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "save_result_path": save_result_path,
    }
    if task_mode == "i2i":
        payload["image_path"] = image_b64
    else:
        payload["aspect_ratio"] = aspect_ratio
    return payload


def run_single_request(
    *,
    endpoint: str,
    payload: dict,
    timeout_s: int,
    poll_interval_s: float,
    decoder_only: bool,
    dec_url: str,
    enc_url: str = "",
    trans_url: str = "",
    disagg_transformer_idx: Optional[int] = None,
    disagg_phase1_room: Optional[int] = None,
    disagg_phase2_room: Optional[int] = None,
) -> str:
    """Run one end-to-end disagg request; returns Decoder task_id.

    Official 3-way order: Decoder -> Transformer -> Encoder, then poll Decoder only.
    """
    payload2 = dict(payload)
    if disagg_transformer_idx is not None:
        payload2["disagg_transformer_idx"] = int(disagg_transformer_idx)
    if disagg_phase1_room is not None:
        payload2["disagg_phase1_room"] = int(disagg_phase1_room)
    if disagg_phase2_room is not None:
        payload2["disagg_phase2_room"] = int(disagg_phase2_room)

    if decoder_only:
        task_id = post_create_task(dec_url, endpoint, payload2)
        _ = poll_task(dec_url, task_id, timeout_s=timeout_s, interval_s=poll_interval_s)
        return task_id

    dec_task_id = post_create_task(dec_url, endpoint, payload2)
    _ = post_create_task(trans_url, endpoint, payload2)
    _ = post_create_task(enc_url, endpoint, payload2)
    _ = poll_task(dec_url, dec_task_id, timeout_s=timeout_s, interval_s=poll_interval_s)
    return dec_task_id


@dataclass
class SustainedJob:
    index: int
    payload: dict
    enqueue_time: float


@dataclass
class SustainedResult:
    ok: bool
    enqueue_time: float
    start_time: float
    done_time: float
    task_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def queue_wait_s(self) -> float:
        return self.start_time - self.enqueue_time

    @property
    def service_s(self) -> float:
        return self.done_time - self.start_time

    @property
    def e2e_s(self) -> float:
        return self.done_time - self.enqueue_time


def run_one_task(
    *,
    endpoint: str,
    payload: dict,
    timeout_s: int,
    poll_interval_s: float,
    out_list: list[TaskResult],
    lock: threading.Lock,
    decoder_only: bool,
    dec_url: str,
    enc_url: str = "",
    trans_url: str = "",
    disagg_transformer_idx: Optional[int] = None,
    disagg_phase1_room: Optional[int] = None,
    disagg_phase2_room: Optional[int] = None,
) -> None:
    submit_time = time.time()
    try:
        task_id = run_single_request(
            endpoint=endpoint,
            payload=payload,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            decoder_only=decoder_only,
            dec_url=dec_url,
            enc_url=enc_url,
            trans_url=trans_url,
            disagg_transformer_idx=disagg_transformer_idx,
            disagg_phase1_room=disagg_phase1_room,
            disagg_phase2_room=disagg_phase2_room,
        )
        done_time = time.time()
        tr = TaskResult(ok=True, submit_time=submit_time, done_time=done_time, task_id=task_id)
    except Exception as e:
        done_time = time.time()
        tr = TaskResult(ok=False, submit_time=submit_time, done_time=done_time, task_id=None, error=str(e))
    with lock:
        out_list.append(tr)


def run_bench(
    *,
    dec_urls: list[str],
    replica_triples: Optional[list[tuple[str, str, str]]],
    decoder_only: bool,
    endpoint: str,
    concurrency: int,
    timeout_s: int,
    poll_interval_s: float,
    task_mode: str,
    img_b64: Any | str | None,
    aspect_ratio: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    save_dir: str,
    run_id: str,
    phase1_room_base: int,
    phase2_room_base: int,
    phase_room_stride: int,
) -> tuple[int, int, int, float, float, float, float, float, float, float, float]:
    results: list[TaskResult] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    t0 = time.time()
    for i in range(concurrency):
        if decoder_only:
            dec_url = dec_urls[i % len(dec_urls)]
            task_kw: dict[str, Any] = {
                "decoder_only": True,
                "dec_url": dec_url,
                "enc_url": "",
                "trans_url": "",
                "disagg_transformer_idx": None,
                "disagg_phase1_room": None,
                "disagg_phase2_room": None,
            }
        else:
            assert replica_triples is not None
            ridx = i % len(replica_triples)
            enc_url, trans_url, dec_url = replica_triples[ridx]
            # Each transformer stack uses its own bootstrap room to avoid waiting_pool overwrite.
            p1_room = phase1_room_base + ridx * phase_room_stride
            p2_room = phase2_room_base + ridx * phase_room_stride
            task_kw = {
                "decoder_only": False,
                "dec_url": dec_url,
                "enc_url": enc_url,
                "trans_url": trans_url,
                "disagg_transformer_idx": ridx,
                "disagg_phase1_room": p1_room,
                "disagg_phase2_room": p2_room,
            }
        payload = make_task_payload(
            task_mode=task_mode,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed + i,
            save_result_path=os.path.join(save_dir, f"disagg_{run_id}_c{concurrency}_{i}.png"),
            image_b64=img_b64,
            aspect_ratio=aspect_ratio,
        )
        th = threading.Thread(
            target=run_one_task,
            kwargs={
                "endpoint": endpoint,
                "payload": payload,
                "timeout_s": timeout_s,
                "poll_interval_s": poll_interval_s,
                "out_list": results,
                "lock": lock,
                **task_kw,
            },
            daemon=False,
        )
        th.start()
        threads.append(th)

    nr = len(replica_triples) if (replica_triples is not None and not decoder_only) else len(dec_urls)
    logger.info(
        f"run_bench: {concurrency} threads started (each runs one full request: 3x POST + poll). "
        f"Replica routing: worker index i -> stack (i % {nr}); decoder base URLs count={len(dec_urls)}. "
        f"Parallel across stacks needs concurrency>1 and nr>1; one stack queues concurrent clients on server."
    )

    for th in threads:
        th.join()
    t1 = time.time()

    oks = [r for r in results if r.ok]
    fails = [r for r in results if not r.ok]
    latencies = [r.latency_s for r in oks]

    if results:
        submit_min = min(r.submit_time for r in results)
        done_max = max(r.done_time for r in results)
        wall_s = done_max - submit_min
    else:
        wall_s = float("nan")

    qps = (len(oks) / wall_s) if wall_s and wall_s > 0 else 0.0
    p50 = percentile(latencies, 0.50) if latencies else float("nan")
    p95 = percentile(latencies, 0.95) if latencies else float("nan")
    p99 = percentile(latencies, 0.99) if latencies else float("nan")
    lat_min = min(latencies) if latencies else float("nan")
    lat_max = max(latencies) if latencies else float("nan")
    return (concurrency, len(oks), len(fails), wall_s, (t1 - t0), qps, p50, p95, p99, lat_min, lat_max)


def run_bench_sustained(
    *,
    dec_urls: list[str],
    replica_triples: Optional[list[tuple[str, str, str]]],
    decoder_only: bool,
    endpoint: str,
    concurrency: int,
    timeout_s: int,
    poll_interval_s: float,
    task_mode: str,
    img_b64: Any | str | None,
    aspect_ratio: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    save_dir: str,
    run_id: str,
    total_requests: int,
    duration_s: float,
    request_interval_s: float,
    queue_maxsize: int,
    phase1_room_base: int,
    phase2_room_base: int,
    phase_room_stride: int,
) -> None:
    """
    Producer-consumer sustained mode:
    - producer creates jobs continuously
    - consumers run full 3-way disagg (or decoder_only) round-robin across replicas
    """
    job_queue: Queue[SustainedJob] = Queue(maxsize=queue_maxsize)
    results: list[SustainedResult] = []
    lock = threading.Lock()

    producer_done = threading.Event()
    enqueued = 0
    dropped = 0
    counter = 0
    start = time.time()

    def producer() -> None:
        nonlocal enqueued, dropped, counter
        while counter < total_requests and (time.time() - start) < duration_s:
            payload = make_task_payload(
                task_mode=task_mode,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed + counter,
                save_result_path=os.path.join(save_dir, f"disagg_sustained_{run_id}_{counter}.png"),
                image_b64=img_b64,
                aspect_ratio=aspect_ratio,
            )
            job = SustainedJob(index=counter, payload=payload, enqueue_time=time.time())
            counter += 1
            try:
                job_queue.put(job, timeout=1.0)
                enqueued += 1
            except Exception:
                dropped += 1
            if request_interval_s > 0:
                time.sleep(request_interval_s)
        producer_done.set()

    def consumer() -> None:
        while True:
            if producer_done.is_set() and job_queue.empty():
                return
            try:
                job = job_queue.get(timeout=0.5)
            except Empty:
                continue
            start_time = time.time()
            task_id = None
            try:
                if decoder_only:
                    ridx = job.index % len(dec_urls)
                    task_id = run_single_request(
                        endpoint=endpoint,
                        payload=job.payload,
                        timeout_s=timeout_s,
                        poll_interval_s=poll_interval_s,
                        decoder_only=True,
                        dec_url=dec_urls[ridx],
                    )
                else:
                    assert replica_triples is not None
                    ridx = job.index % len(replica_triples)
                    enc_url, trans_url, dec_url = replica_triples[ridx]
                    p1_room = phase1_room_base + ridx * phase_room_stride
                    p2_room = phase2_room_base + ridx * phase_room_stride
                    task_id = run_single_request(
                        endpoint=endpoint,
                        payload=job.payload,
                        timeout_s=timeout_s,
                        poll_interval_s=poll_interval_s,
                        decoder_only=False,
                        dec_url=dec_url,
                        enc_url=enc_url,
                        trans_url=trans_url,
                        disagg_transformer_idx=ridx,
                        disagg_phase1_room=p1_room,
                        disagg_phase2_room=p2_room,
                    )
                done_time = time.time()
                item = SustainedResult(True, job.enqueue_time, start_time, done_time, task_id=task_id)
            except Exception as e:
                done_time = time.time()
                item = SustainedResult(False, job.enqueue_time, start_time, done_time, task_id=task_id, error=str(e))
            with lock:
                results.append(item)
            job_queue.task_done()

    workers = [threading.Thread(target=consumer, daemon=False) for _ in range(concurrency)]
    t0 = time.time()
    prod = threading.Thread(target=producer, daemon=False)
    prod.start()
    for w in workers:
        w.start()

    nr = len(replica_triples) if (replica_triples is not None and not decoder_only) else len(dec_urls)
    logger.info(
        f"run_sustained: {concurrency} consumer threads + producer; job routing: job.index -> stack (i % {nr}). "
        f"Same parallelism notes as run_bench (multi-stack needs nr>1 and enough concurrent consumers)."
    )

    prod.join()
    job_queue.join()
    for w in workers:
        w.join()
    t1 = time.time()

    oks = [r for r in results if r.ok]
    fails = [r for r in results if not r.ok]
    if results:
        start_min = min(r.enqueue_time for r in results)
        done_max = max(r.done_time for r in results)
        wall_s = done_max - start_min
    else:
        wall_s = float("nan")

    qps = (len(oks) / wall_s) if wall_s and wall_s > 0 else 0.0

    e2e = [r.e2e_s for r in oks]
    queue_wait = [r.queue_wait_s for r in oks]
    service = [r.service_s for r in oks]

    def fmt(x: float) -> float:
        return float(x)

    e2e_p50 = percentile(e2e, 0.50) if e2e else float("nan")
    e2e_p95 = percentile(e2e, 0.95) if e2e else float("nan")
    e2e_p99 = percentile(e2e, 0.99) if e2e else float("nan")
    queue_p50 = percentile(queue_wait, 0.50) if queue_wait else float("nan")
    queue_p95 = percentile(queue_wait, 0.95) if queue_wait else float("nan")
    queue_p99 = percentile(queue_wait, 0.99) if queue_wait else float("nan")
    svc_p50 = percentile(service, 0.50) if service else float("nan")
    svc_p95 = percentile(service, 0.95) if service else float("nan")
    svc_p99 = percentile(service, 0.99) if service else float("nan")

    lat_min = min(e2e) if e2e else float("nan")
    lat_max = max(e2e) if e2e else float("nan")

    logger.info(
        f"[sustained] workers={concurrency} enqueued={enqueued} dropped={dropped} ok={len(oks)} failed={len(fails)} "
        f"wall_s={wall_s:.2f} elapsed_s={(t1 - t0):.2f} qps={qps:.4f}"
    )
    logger.info(
        f"[sustained] e2e(s): p50={fmt(e2e_p50):.2f} p95={fmt(e2e_p95):.2f} p99={fmt(e2e_p99):.2f} min={fmt(lat_min):.2f} max={fmt(lat_max):.2f}"
    )
    logger.info(
        f"[sustained] queue_wait(s): p50={queue_p50:.2f} p95={queue_p95:.2f} p99={queue_p99:.2f}; service(s): p50={svc_p50:.2f} p95={svc_p95:.2f} p99={svc_p99:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Qwen Image 3-way disagg bench: default POST order Decoder->Transformer->Encoder, poll Decoder (see DisaggSplitDeploy.md).",
    )
    ap.add_argument(
        "--task",
        type=str,
        choices=("t2i", "i2i"),
        default="t2i",
        help="t2i matches start_multi_server_disagg.sh (text-only payload); i2i requires --image_path.",
    )
    ap.add_argument(
        "--url",
        type=str,
        default="",
        help="Single Decoder base URL. Default: http://127.0.0.1:8008 (t2i) or http://127.0.0.1:8014 (i2i) if --replicas not set.",
    )
    ap.add_argument(
        "--url_list",
        type=str,
        default="",
        help="Comma-separated Decoder URLs (round-robin). Mutually exclusive with --replicas.",
    )
    ap.add_argument(
        "--trans_url_list",
        type=str,
        default="",
        help="Comma-separated Transformer URLs. When set (and not --decoder_only), enables shared-ED + multi-transformer routing.",
    )
    ap.add_argument(
        "--replicas",
        type=int,
        default=0,
        help="If >0, build Decoder URLs like start_multi_server_disagg.sh: dec_port = dec_port_base + i*port_step.",
    )
    ap.add_argument(
        "--dec_host",
        type=str,
        default="127.0.0.1",
        help="Host for --replicas URL expansion (default 127.0.0.1).",
    )
    ap.add_argument(
        "--dec_port_base",
        type=int,
        default=8008,
        help="Decoder port for replica 0 (default 8008, same as shell DEC_PORT_BASE).",
    )
    ap.add_argument(
        "--port_step",
        type=int,
        default=10,
        help="Port increment per replica (default 10, same as shell PORT_STEP).",
    )
    ap.add_argument(
        "--enc_port_base",
        type=int,
        default=8002,
        help="Encoder HTTP port for replica 0 (default 8002; I2I example often 8012).",
    )
    ap.add_argument(
        "--trans_port_base",
        type=int,
        default=8003,
        help="Transformer HTTP port for replica 0 (default 8003; I2I example often 8013).",
    )
    ap.add_argument(
        "--decoder_only",
        action="store_true",
        help="Only POST to Decoder (Encoder/Transformer receive no HTTP; normal 3-way will not finish).",
    )
    ap.add_argument(
        "--endpoint",
        type=str,
        default="/v1/tasks/image/",
        help="POST endpoint on each stage (same path on Decoder, Transformer, Encoder).",
    )
    ap.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests.")
    ap.add_argument(
        "--concurrency_list",
        type=str,
        default="1,2,4,8",
        help="Comma-separated concurrencies to run in one shot. Empty to disable and use --concurrency only.",
    )
    ap.add_argument("--timeout_s", type=int, default=3600, help="Per-task timeout in seconds.")
    ap.add_argument("--poll_interval_s", type=float, default=1.0, help="Polling interval in seconds.")
    ap.add_argument(
        "--image_path",
        type=str,
        default="",
        help="I2I only: local image path, HTTP URL, or empty for t2i.",
    )
    ap.add_argument(
        "--aspect_ratio",
        type=str,
        default="16:9",
        help="T2I only: passed as aspect_ratio in JSON (default 16:9).",
    )
    ap.add_argument("--prompt", type=str, default="A cat wearing sunglasses and working as a lifeguard at pool.")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="/tmp/lightx2v_bench_outputs", help="Unique save paths per request.")
    ap.add_argument("--sustained", action="store_true", help="Enable sustained mode (producer-consumer queue).")
    ap.add_argument("--total_requests", type=int, default=200, help="Max requests to generate in sustained mode.")
    ap.add_argument("--duration_s", type=float, default=300.0, help="Max generate duration in sustained mode.")
    ap.add_argument(
        "--request_interval_s",
        type=float,
        default=0.0,
        help="Producer interval between requests (0 means as fast as possible).",
    )
    ap.add_argument("--queue_maxsize", type=int, default=512, help="Bounded request queue size.")
    ap.add_argument("--phase1_room_base", type=int, default=1000000, help="bootstrap_room base for Phase1 (Encoder->Transformer) per transformer.")
    ap.add_argument("--phase2_room_base", type=int, default=2000000, help="decoder_bootstrap_room base for Phase2 (Transformer->Decoder) per transformer.")
    ap.add_argument("--phase_room_stride", type=int, default=100000, help="room id stride per transformer index r.")
    args = ap.parse_args()

    if args.task == "i2i" and not (args.image_path or "").strip():
        raise SystemExit("[ERROR] --task i2i requires a non-empty --image_path.")

    os.makedirs(args.save_dir, exist_ok=True)
    img_b64: Any | str | None
    if args.task == "i2i":
        img_b64 = process_image_path(args.image_path)
    else:
        img_b64 = None

    run_id = uuid.uuid4().hex[:8]

    explicit_urls = parse_url_list(args.url_list)
    explicit_trans_urls = parse_url_list(args.trans_url_list)
    if args.replicas > 0 and explicit_urls:
        raise SystemExit("[ERROR] Use either --replicas or --url_list, not both.")
    if args.replicas > 0:
        dec_urls = build_decoder_urls_from_replicas(
            args.dec_host,
            args.dec_port_base,
            args.port_step,
            args.replicas,
        )
    elif explicit_urls:
        dec_urls = explicit_urls
    else:
        default_url = args.url.strip() or (
            "http://127.0.0.1:8008" if args.task == "t2i" else "http://127.0.0.1:8014"
        )
        dec_urls = [default_url]

    replica_triples: Optional[list[tuple[str, str, str]]] = None
    if not args.decoder_only:
        if explicit_trans_urls:
            if not dec_urls:
                raise SystemExit("[ERROR] shared-ED mode needs one decoder URL from --url or --url_list.")
            if len(dec_urls) != 1:
                logger.warning(
                    f"shared-ED mode expects one decoder URL; got {len(dec_urls)} in --url_list, using first: {dec_urls[0]}"
                )
            replica_triples = build_shared_ed_groups(
                dec_url=dec_urls[0],
                enc_port_base=args.enc_port_base,
                transformer_urls=explicit_trans_urls,
            )
        elif args.replicas > 0:
            replica_triples = build_disagg_triples_from_replicas(
                args.dec_host,
                args.enc_port_base,
                args.trans_port_base,
                args.dec_port_base,
                args.port_step,
                args.replicas,
            )
        else:
            replica_triples = build_disagg_triples_from_decoder_urls(
                dec_urls,
                args.enc_port_base,
                args.trans_port_base,
                args.port_step,
            )

    if args.sustained:
        run_bench_sustained(
            dec_urls=dec_urls,
            replica_triples=replica_triples,
            decoder_only=args.decoder_only,
            endpoint=args.endpoint,
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval_s,
            task_mode=args.task,
            img_b64=img_b64,
            aspect_ratio=args.aspect_ratio,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            save_dir=args.save_dir,
            run_id=run_id,
            total_requests=args.total_requests,
            duration_s=args.duration_s,
            request_interval_s=args.request_interval_s,
            queue_maxsize=args.queue_maxsize,
            phase1_room_base=args.phase1_room_base,
            phase2_room_base=args.phase2_room_base,
            phase_room_stride=args.phase_room_stride,
        )
        return

    conc_list = parse_concurrency_list(args.concurrency_list)
    if not conc_list:
        conc_list = [args.concurrency]

    mode = "decoder_only" if args.decoder_only else "three_way(Decoder->Transformer->Encoder)"
    if explicit_trans_urls and not args.decoder_only:
        mode = "three_way(shared ED + multi-transformer)"
    logger.info(
        f"Disagg bench mode={mode} task={args.task} decoders={dec_urls} endpoint={args.endpoint} "
        f"concurrency_list={conc_list} timeout_s={args.timeout_s} poll_interval_s={args.poll_interval_s}"
    )
    if explicit_trans_urls and not args.decoder_only:
        logger.info(f"Transformers (round-robin): {explicit_trans_urls}")
    logger.info("Summary columns: N ok failed wall_s elapsed_s qps p50 p95 p99 min max")

    for n in conc_list:
        (n, ok, failed, wall_s, elapsed_s, qps, p50, p95, p99, lat_min, lat_max) = run_bench(
            dec_urls=dec_urls,
            replica_triples=replica_triples,
            decoder_only=args.decoder_only,
            endpoint=args.endpoint,
            concurrency=n,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval_s,
            task_mode=args.task,
            img_b64=img_b64,
            aspect_ratio=args.aspect_ratio,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            save_dir=args.save_dir,
            run_id=run_id,
            phase1_room_base=args.phase1_room_base,
            phase2_room_base=args.phase2_room_base,
            phase_room_stride=args.phase_room_stride,
        )
        logger.info(
            f"N={n} ok={ok} failed={failed} wall_s={wall_s:.2f} elapsed_s={elapsed_s:.2f} "
            f"qps={qps:.4f} p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} min={lat_min:.2f} max={lat_max:.2f}"
        )


if __name__ == "__main__":
    main()
