import argparse
import base64
import os
import threading
import time
import uuid
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


def run_single_request(
    *,
    base_url: str,
    endpoint: str,
    payload: dict,
    timeout_s: int,
    poll_interval_s: float,
) -> str:
    """POST one task to base_url, then poll until completion. Returns task_id."""
    resp = requests.post(f"{base_url}{endpoint}", json=payload, timeout=30)
    resp.raise_for_status()
    task_id = resp.json().get("task_id")
    if not task_id:
        raise RuntimeError(f"No task_id in response: {resp.text}")
    _ = poll_task(base_url, task_id, timeout_s=timeout_s, interval_s=poll_interval_s)
    return task_id


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
    base_url: str,
    endpoint: str,
    payload: dict,
    timeout_s: int,
    poll_interval_s: float,
    out_list: list[TaskResult],
    lock: threading.Lock,
) -> None:
    submit_time = time.time()
    try:
        resp = requests.post(f"{base_url}{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        task_id = resp.json().get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in response: {resp.text}")
        _ = poll_task(base_url, task_id, timeout_s=timeout_s, interval_s=poll_interval_s)
        done_time = time.time()
        tr = TaskResult(ok=True, submit_time=submit_time, done_time=done_time, task_id=task_id)
    except Exception as e:
        done_time = time.time()
        tr = TaskResult(ok=False, submit_time=submit_time, done_time=done_time, task_id=None, error=str(e))
    with lock:
        out_list.append(tr)


def run_bench(
    *,
    url_list: list[str],
    endpoint: str,
    concurrency: int,
    timeout_s: int,
    poll_interval_s: float,
    img_b64: Any | str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    save_dir: str,
    run_id: str,
) -> tuple[int, int, int, float, float, float, float, float, float]:
    results: list[TaskResult] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    t0 = time.time()
    for i in range(concurrency):
        base_url = url_list[i % len(url_list)]
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_path": img_b64,
            "seed": seed + i,
            "save_result_path": os.path.join(save_dir, f"baseline_{run_id}_c{concurrency}_{i}.mp4"),
        }
        th = threading.Thread(
            target=run_one_task,
            kwargs={
                "base_url": base_url,
                "endpoint": endpoint,
                "payload": payload,
                "timeout_s": timeout_s,
                "poll_interval_s": poll_interval_s,
                "out_list": results,
                "lock": lock,
            },
            daemon=False,
        )
        th.start()
        threads.append(th)

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
    url_list: list[str],
    endpoint: str,
    concurrency: int,
    timeout_s: int,
    poll_interval_s: float,
    img_b64: Any | str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    save_dir: str,
    run_id: str,
    total_requests: int,
    duration_s: float,
    request_interval_s: float,
    queue_maxsize: int,
) -> None:
    """
    Producer-consumer sustained mode:
    - producer creates jobs continuously
    - consumers (workers) dispatch jobs to base_url in round-robin
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
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_path": img_b64,
                "seed": seed + counter,
                "save_result_path": os.path.join(save_dir, f"sustained_{run_id}_{counter}.mp4"),
            }
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
                base_url = url_list[job.index % len(url_list)]
                task_id = run_single_request(
                    base_url=base_url,
                    endpoint=endpoint,
                    payload=job.payload,
                    timeout_s=timeout_s,
                    poll_interval_s=poll_interval_s,
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
    ap = argparse.ArgumentParser(description="Concurrent benchmark for baseline (single service).")
    ap.add_argument("--url", type=str, default="http://localhost:8005", help="Baseline server base URL.")
    ap.add_argument(
        "--url_list",
        type=str,
        default="",
        help="Comma-separated baseline server base URLs to dispatch across (e.g. 4-card). If empty, use --url.",
    )
    ap.add_argument("--endpoint", type=str, default="/v1/tasks/video/", help="POST endpoint for video tasks.")
    ap.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests.")
    ap.add_argument(
        "--concurrency_list",
        type=str,
        default="1,2,4,8",
        help="Comma-separated concurrencies to run in one shot. Empty to disable and use --concurrency only.",
    )
    ap.add_argument("--timeout_s", type=int, default=3600, help="Per-task timeout in seconds.")
    ap.add_argument("--poll_interval_s", type=float, default=1.0, help="Polling interval in seconds.")
    ap.add_argument("--image_path", type=str, default="assets/inputs/imgs/img_0.jpg", help="Local image path (will be base64-encoded) or HTTP URL.")
    ap.add_argument("--prompt", type=str, default="A cat wearing sunglasses and working as a lifeguard at pool.")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="/tmp/lightx2v_bench_outputs", help="Unique save paths per request.")
    ap.add_argument("--sustained", action="store_true", help="Enable sustained mode (producer-consumer queue).")
    ap.add_argument("--total_requests", type=int, default=200, help="Max requests to generate in sustained mode.")
    ap.add_argument("--duration_s", type=float, default=300.0, help="Max generate duration in sustained mode.")
    ap.add_argument("--request_interval_s", type=float, default=0.0, help="Producer interval between requests (0 means as fast as possible).")
    ap.add_argument("--queue_maxsize", type=int, default=512, help="Bounded request queue size.")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    img_b64 = process_image_path(args.image_path)

    run_id = uuid.uuid4().hex[:8]

    url_list = parse_url_list(args.url_list)
    if not url_list:
        url_list = [args.url]

    if args.sustained:
        run_bench_sustained(
            url_list=url_list,
            endpoint=args.endpoint,
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval_s,
            img_b64=img_b64,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            save_dir=args.save_dir,
            run_id=run_id,
            total_requests=args.total_requests,
            duration_s=args.duration_s,
            request_interval_s=args.request_interval_s,
            queue_maxsize=args.queue_maxsize,
        )
        return

    conc_list = parse_concurrency_list(args.concurrency_list)
    if not conc_list:
        conc_list = [args.concurrency]

    logger.info(
        f"Baseline concurrent bench: url_list={url_list} endpoint={args.endpoint} "
        f"concurrency_list={conc_list} timeout_s={args.timeout_s} poll_interval_s={args.poll_interval_s}"
    )
    logger.info("Summary columns: N ok failed wall_s elapsed_s qps p50 p95 p99 min max")

    for n in conc_list:
        (n, ok, failed, wall_s, elapsed_s, qps, p50, p95, p99, lat_min, lat_max) = run_bench(
            url_list=url_list,
            endpoint=args.endpoint,
            concurrency=n,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval_s,
            img_b64=img_b64,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            save_dir=args.save_dir,
            run_id=run_id,
        )
        logger.info(
            f"N={n} ok={ok} failed={failed} wall_s={wall_s:.2f} elapsed_s={elapsed_s:.2f} "
            f"qps={qps:.4f} p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} min={lat_min:.2f} max={lat_max:.2f}"
        )


if __name__ == "__main__":
    main()