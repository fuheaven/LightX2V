"""Opt-in, low-overhead communication profiling for benchmark runs.

Set ``LIGHTX2V_COMM_PROFILE=1`` to record CUDA-event timings for the
distributed collectives explicitly wrapped by this module.  Events are
resolved once at the end of a request so the profiler does not insert a
device-wide synchronization around every collective.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.distributed as dist
from loguru import logger


_LOCK = threading.Lock()
_ACTIVE = False
_REQUEST_ID: str | int | None = None
_RECORDS: list[dict[str, Any]] = []


def _env_enabled(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def start_communication_profile(request_id: str | int | None = None) -> bool:
    """Start a request-scoped profile when the opt-in environment flag is set."""
    if not _env_enabled("LIGHTX2V_COMM_PROFILE"):
        return False

    global _ACTIVE, _REQUEST_ID, _RECORDS
    with _LOCK:
        _ACTIVE = True
        _REQUEST_ID = request_id if request_id is not None else os.getenv("LIGHTX2V_PROFILE_REQUEST_ID")
        _RECORDS = []
    return True


def communication_profile_active() -> bool:
    return _ACTIVE


@contextmanager
def cuda_communication_region(
    name: str,
    *,
    tx_bytes: int = 0,
    rx_bytes: int = 0,
) -> Iterator[None]:
    """Record one stream-ordered communication region without synchronizing it."""
    if not _ACTIVE:
        yield
        return

    start_event = None
    end_event = None
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    wall_start = time.perf_counter()
    try:
        yield
    finally:
        wall_submit_s = time.perf_counter() - wall_start
        if end_event is not None:
            end_event.record()
        record = {
            "name": str(name),
            "tx_bytes": int(tx_bytes),
            "rx_bytes": int(rx_bytes),
            "wall_submit_s": float(wall_submit_s),
            "start_event": start_event,
            "end_event": end_event,
        }
        with _LOCK:
            if _ACTIVE:
                _RECORDS.append(record)


def profiled_all_gather(
    output_tensors,
    input_tensor: torch.Tensor,
    *,
    group=None,
    name: str,
    async_op: bool = False,
):
    """Profile a tensor all-gather and retain its logical per-rank traffic."""
    world_size = dist.get_world_size(group)
    peer_bytes = int(input_tensor.numel() * input_tensor.element_size()) * max(0, world_size - 1)
    with cuda_communication_region(name, tx_bytes=peer_bytes, rx_bytes=peer_bytes):
        return dist.all_gather(output_tensors, input_tensor, group=group, async_op=async_op)


def finish_communication_profile() -> dict[str, Any] | None:
    """Resolve recorded events, log one machine-readable summary, and reset."""
    global _ACTIVE, _REQUEST_ID, _RECORDS
    with _LOCK:
        if not _ACTIVE:
            return None
        records = list(_RECORDS)
        request_id = _REQUEST_ID
        _ACTIVE = False
        _REQUEST_ID = None
        _RECORDS = []

    if records and torch.cuda.is_available():
        torch.cuda.synchronize()

    grouped: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "count": 0,
            "gpu_time_s": 0.0,
            "wall_submit_s": 0.0,
            "tx_bytes": 0,
            "rx_bytes": 0,
        }
    )
    for record in records:
        gpu_time_s = 0.0
        start_event = record.get("start_event")
        end_event = record.get("end_event")
        if start_event is not None and end_event is not None:
            gpu_time_s = float(start_event.elapsed_time(end_event)) / 1000.0
        bucket = grouped[str(record["name"])]
        bucket["count"] = int(bucket["count"]) + 1
        bucket["gpu_time_s"] = float(bucket["gpu_time_s"]) + gpu_time_s
        bucket["wall_submit_s"] = float(bucket["wall_submit_s"]) + float(record["wall_submit_s"])
        bucket["tx_bytes"] = int(bucket["tx_bytes"]) + int(record["tx_bytes"])
        bucket["rx_bytes"] = int(bucket["rx_bytes"]) + int(record["rx_bytes"])

    by_collective = {}
    for name, values in sorted(grouped.items()):
        by_collective[name] = {
            "count": int(values["count"]),
            "gpu_time_s": float(values["gpu_time_s"]),
            "wall_submit_s": float(values["wall_submit_s"]),
            "tx_bytes": int(values["tx_bytes"]),
            "rx_bytes": int(values["rx_bytes"]),
        }

    summary = {
        "schema_version": 1,
        "request_id": request_id,
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "communication_time_s": sum(float(item["gpu_time_s"]) for item in by_collective.values()),
        "tx_bytes": sum(int(item["tx_bytes"]) for item in by_collective.values()),
        "rx_bytes": sum(int(item["rx_bytes"]) for item in by_collective.values()),
        "by_collective": by_collective,
    }
    logger.info("[CommProfile] {}", json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return summary
