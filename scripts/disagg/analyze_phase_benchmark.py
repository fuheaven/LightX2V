#!/usr/bin/env python3
"""Build a reproducible phase/communication analysis for Wan2.2 benchmarks."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PROFILE_RE = re.compile(
    r"\[Profile\] Rank (?P<rank>\d+) - .*? "
    r"(?P<name>Run VAE Encoder|Run Text Encoder|Run Dit every step|Run VAE Decoder|RUN pipeline) "
    r"cost (?P<seconds>[0-9.]+) seconds"
)
COMM_MARKER = "[CommProfile] "
TRANSFER_RE = re.compile(
    r"\[DataTransferProfile\] phase=(?P<phase>\S+) mode=(?P<mode>\S+) "
    r"room=(?P<room>\d+) bytes=(?P<bytes>\d+) duration_s=(?P<seconds>[0-9.]+) "
    r"status=(?P<status>-?\d+)"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _range(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": _mean(values),
        "median": _median(values),
        "min": min(values),
        "max": max(values),
    }


def _parse_ranked_baseline(path: Path) -> dict[str, Any]:
    phase_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    comm_by_rank: dict[int, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            profile_match = PROFILE_RE.search(line)
            if profile_match:
                phase_values[int(profile_match.group("rank"))][profile_match.group("name")].append(float(profile_match.group("seconds")))
                continue
            marker_index = line.find(COMM_MARKER)
            if marker_index >= 0:
                payload = json.loads(line[marker_index + len(COMM_MARKER) :])
                comm_by_rank[int(payload["rank"])] = payload

    ranks = sorted(set(phase_values) | set(comm_by_rank))
    if ranks != list(range(8)):
        raise ValueError(f"expected ranked baseline data for ranks 0..7, got {ranks}")

    rows = []
    for rank in ranks:
        phases = phase_values[rank]
        required_counts = {
            "Run Text Encoder": 1,
            "Run VAE Encoder": 1,
            "Run Dit every step": 4,
            "Run VAE Decoder": 1,
            "RUN pipeline": 1,
        }
        for name, expected_count in required_counts.items():
            if len(phases.get(name, [])) != expected_count:
                raise ValueError(f"rank {rank}: expected {expected_count} '{name}' entries, got {len(phases.get(name, []))}")

        comm = comm_by_rank[rank]
        collectives = comm.get("by_collective", {})
        dit_comm_s = sum(
            float(collectives.get(name, {}).get("gpu_time_s", 0.0))
            for name in ("dit_ulysses_all_to_all", "dit_sequence_output_all_gather")
        )
        vae_encoder_comm_s = float(collectives.get("vae_encoder_all_gather", {}).get("gpu_time_s", 0.0))
        vae_decoder_comm_s = float(collectives.get("vae_decoder_all_gather", {}).get("gpu_time_s", 0.0))
        text_s = phases["Run Text Encoder"][0]
        vae_encoder_s = phases["Run VAE Encoder"][0]
        dit_s = sum(phases["Run Dit every step"])
        vae_decoder_s = phases["Run VAE Decoder"][0]
        pipeline_s = phases["RUN pipeline"][0]
        row = {
            "rank": rank,
            "pipeline_s": pipeline_s,
            "text_encoder_s": text_s,
            "vae_encoder_s": vae_encoder_s,
            "dit_s": dit_s,
            "vae_decoder_s": vae_decoder_s,
            "communication_s": float(comm.get("communication_time_s", 0.0)),
            "dit_communication_s": dit_comm_s,
            "vae_encoder_communication_s": vae_encoder_comm_s,
            "vae_decoder_communication_s": vae_decoder_comm_s,
            "tx_bytes": int(comm.get("tx_bytes", 0)),
            "rx_bytes": int(comm.get("rx_bytes", 0)),
        }
        row["stage_residual_s"] = pipeline_s - text_s - vae_encoder_s - dit_s - vae_decoder_s
        rows.append(row)

    critical = max(rows, key=lambda item: item["pipeline_s"])
    categories = [
        {
            "category": "Text encoder compute",
            "seconds": critical["text_encoder_s"],
        },
        {
            "category": "VAE encoder compute excluding collectives",
            "seconds": max(0.0, critical["vae_encoder_s"] - critical["vae_encoder_communication_s"]),
        },
        {
            "category": "VAE encoder communication",
            "seconds": critical["vae_encoder_communication_s"],
        },
        {
            "category": "DiT compute excluding collectives",
            "seconds": max(0.0, critical["dit_s"] - critical["dit_communication_s"]),
        },
        {
            "category": "DiT communication",
            "seconds": critical["dit_communication_s"],
        },
        {
            "category": "VAE decoder compute excluding collectives",
            "seconds": max(0.0, critical["vae_decoder_s"] - critical["vae_decoder_communication_s"]),
        },
        {
            "category": "VAE decoder communication",
            "seconds": critical["vae_decoder_communication_s"],
        },
        {
            "category": "Pipeline residual / host I/O",
            "seconds": critical["stage_residual_s"],
        },
    ]
    for item in categories:
        item["share"] = item["seconds"] / critical["pipeline_s"]

    return {
        "rank_rows": rows,
        "critical_rank": critical["rank"],
        "critical_rank_pipeline_s": critical["pipeline_s"],
        "critical_rank_categories": categories,
        "communication_s_across_ranks": _range([row["communication_s"] for row in rows]),
        "communication_share_of_pipeline_critical_rank": critical["communication_s"] / critical["pipeline_s"],
        "dit_communication_share_critical_rank": critical["dit_communication_s"] / critical["dit_s"],
        "collective_counts_per_rank": {
            name: int(values.get("count", 0))
            for name, values in comm_by_rank[critical["rank"]].get("by_collective", {}).items()
        },
        "logical_traffic_per_rank": {
            "tx_bytes": critical["tx_bytes"],
            "rx_bytes": critical["rx_bytes"],
        },
    }


def _parse_transfer_profiles(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    profiles: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = TRANSFER_RE.search(line)
            if not match:
                continue
            key = (match.group("phase"), int(match.group("room")))
            if key in profiles:
                raise ValueError(f"duplicate transfer profile for {key}")
            profiles[key] = {
                "phase": match.group("phase"),
                "mode": match.group("mode"),
                "room": int(match.group("room")),
                "bytes": int(match.group("bytes")),
                "seconds": float(match.group("seconds")),
                "status": int(match.group("status")),
            }
    return profiles


def _metric(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    return float(value) if value is not None else 0.0


def _build_disagg_analysis(metrics: dict[str, Any], transfer_profiles: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    requests = metrics.get("requests", [])
    if not isinstance(requests, list) or not requests:
        raise ValueError("disagg metrics contain no requests")

    rows = []
    for request in requests:
        request_metrics = request.get("request_metrics", {})
        request_id = int(request_metrics.get("request_id", request.get("data_bootstrap_room", -1)))
        summary = request.get("latency_summary", {})
        phase1_profile = transfer_profiles.get(("phase1", request_id))
        phase2_profile = transfer_profiles.get(("phase2", request_id))
        if phase1_profile is None or phase2_profile is None:
            raise ValueError(f"missing sender-side transfer profile for request {request_id}")
        if phase1_profile["status"] != 0 or phase2_profile["status"] != 0:
            raise ValueError(f"failed sender-side transfer for request {request_id}")

        phase1_visible_s = _metric(summary, "phase1_tensor_transfer_delay_s")
        phase2_visible_s = _metric(summary, "phase2_tensor_transfer_delay_s")
        phase1_wire_s = float(phase1_profile["seconds"])
        phase2_wire_s = float(phase2_profile["seconds"])

        model_compute_s = (
            _metric(summary, "text_encoder_compute_delay_s")
            + _metric(summary, "vae_encoder_compute_delay_s")
            + _metric(summary, "dit_compute_delay_s")
            + _metric(summary, "vae_decoder_compute_delay_s")
        )
        serialization_and_io_s = (
            _metric(summary, "encoder_active_until_enqueue_delay_s")
            - _metric(summary, "text_encoder_compute_delay_s")
            - _metric(summary, "vae_encoder_compute_delay_s")
            + _metric(summary, "transformer_active_until_enqueue_delay_s")
            - _metric(summary, "dit_compute_delay_s")
            + _metric(summary, "decoder_active_until_enqueue_delay_s")
            - _metric(summary, "vae_decoder_compute_delay_s")
        )
        sender_wire_s = phase1_wire_s + phase2_wire_s
        # Sender and receiver scopes use independent clocks and do not have
        # identical boundaries: the receiver may begin observing a transfer
        # after transfer_sync has already started.  Only the receiver-visible
        # portion can be allocated onto the controller's critical path.
        critical_path_wire_s = min(phase1_wire_s, phase1_visible_s) + min(phase2_wire_s, phase2_visible_s)
        transfer_backpressure_s = phase1_visible_s + phase2_visible_s - critical_path_wire_s
        metadata_control_s = (
            _metric(summary, "controller_dispatch_delay_s")
            + _metric(summary, "phase1_metadata_notification_delay_s")
            + _metric(summary, "phase2_metadata_notification_delay_s")
            + _metric(summary, "result_callback_delay_s")
        )
        setup_and_queue_s = (
            _metric(summary, "encoder_queue_delay_s")
            + _metric(summary, "transformer_setup_before_transfer_delay_s")
            + _metric(summary, "transformer_queue_after_transfer_delay_s")
            + _metric(summary, "decoder_setup_before_transfer_delay_s")
            + _metric(summary, "decoder_queue_after_transfer_delay_s")
        )
        e2e_s = _metric(summary, "end_to_end_delay_s")
        category_sum_s = (
            model_compute_s
            + serialization_and_io_s
            + critical_path_wire_s
            + transfer_backpressure_s
            + metadata_control_s
            + setup_and_queue_s
        )
        rows.append(
            {
                "request_id": request_id,
                "e2e_s": e2e_s,
                "text_encoder_s": _metric(summary, "text_encoder_compute_delay_s"),
                "vae_encoder_s": _metric(summary, "vae_encoder_compute_delay_s"),
                "dit_s": _metric(summary, "dit_compute_delay_s"),
                "vae_decoder_s": _metric(summary, "vae_decoder_compute_delay_s"),
                "phase1_bytes": int(phase1_profile["bytes"]),
                "phase1_wire_s": phase1_wire_s,
                "phase1_receiver_visible_s": phase1_visible_s,
                "phase2_bytes": int(phase2_profile["bytes"]),
                "phase2_wire_s": phase2_wire_s,
                "phase2_receiver_visible_s": phase2_visible_s,
                "model_compute_s": model_compute_s,
                "serialization_and_io_s": serialization_and_io_s,
                "sender_wire_s": sender_wire_s,
                "wire_transfer_s": critical_path_wire_s,
                "transfer_backpressure_s": transfer_backpressure_s,
                "metadata_control_s": metadata_control_s,
                "setup_and_queue_s": setup_and_queue_s,
                "category_sum_s": category_sum_s,
                "accounting_delta_s": e2e_s - category_sum_s,
            }
        )

    categories = [
        "model_compute_s",
        "serialization_and_io_s",
        "wire_transfer_s",
        "transfer_backpressure_s",
        "metadata_control_s",
        "setup_and_queue_s",
    ]
    mean_e2e_s = _mean([row["e2e_s"] for row in rows])
    mean_category_rows = [
        {
            "category": category.removesuffix("_s").replace("_", " "),
            "mean_s": _mean([row[category] for row in rows]),
        }
        for category in categories
    ]
    for item in mean_category_rows:
        item["share_of_mean_e2e"] = item["mean_s"] / mean_e2e_s

    steady_rows = [row for row in rows if row["request_id"] > 0] or rows
    active_batch_s = max(row["e2e_s"] for row in rows)
    return {
        "request_rows": rows,
        "active_batch_s": active_batch_s,
        "throughput_per_min": len(rows) * 60.0 / active_batch_s,
        "mean_e2e_s": mean_e2e_s,
        "mean_path_categories": mean_category_rows,
        "steady_stage_seconds": {
            "sample_count": len(steady_rows),
            "text_encoder": _range([row["text_encoder_s"] for row in steady_rows]),
            "vae_encoder": _range([row["vae_encoder_s"] for row in steady_rows]),
            "dit": _range([row["dit_s"] for row in steady_rows]),
            "vae_decoder": _range([row["vae_decoder_s"] for row in steady_rows]),
        },
        "sender_wire_seconds": {
            "phase1": _range([row["phase1_wire_s"] for row in rows]),
            "phase2": _range([row["phase2_wire_s"] for row in rows]),
            "combined": _range([row["sender_wire_s"] for row in rows]),
        },
        "wire_effective_mib_per_s": {
            "phase1": _range([row["phase1_bytes"] / row["phase1_wire_s"] / (1024.0 * 1024.0) for row in rows]),
            "phase2": _range([row["phase2_bytes"] / row["phase2_wire_s"] / (1024.0 * 1024.0) for row in rows]),
        },
        "sender_wire_share_of_mean_e2e": _mean([row["sender_wire_s"] for row in rows]) / mean_e2e_s,
        "critical_path_wire_share_of_mean_e2e": _mean([row["wire_transfer_s"] for row in rows]) / mean_e2e_s,
        "transport_related_visible_share_of_mean_e2e": _mean(
            [row["wire_transfer_s"] + row["transfer_backpressure_s"] + row["metadata_control_s"] for row in rows]
        )
        / mean_e2e_s,
        "max_absolute_accounting_delta_s": max(abs(row["accounting_delta_s"]) for row in rows),
    }


def _baseline_service_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    requests = metrics.get("requests", [])
    if not isinstance(requests, list) or not requests:
        raise ValueError("baseline metrics contain no requests")
    completions = [float(item["elapsed_from_global_start_s"]) for item in requests]
    active_batch_s = max(completions)
    return {
        "request_count": len(requests),
        "active_batch_s": active_batch_s,
        "throughput_per_min": len(requests) * 60.0 / active_batch_s,
        "first_completion_s": min(completions),
        "success_count": sum(int(item.get("return_code", 1)) == 0 for item in requests),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--baseline-ranked-log", type=Path, required=True)
    parser.add_argument("--disagg-metrics", type=Path, required=True)
    parser.add_argument("--disagg-transfer-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline_metrics = _load_json(args.baseline_metrics)
    disagg_metrics = _load_json(args.disagg_metrics)
    baseline_service = _baseline_service_summary(baseline_metrics)
    baseline_ranked = _parse_ranked_baseline(args.baseline_ranked_log)
    transfer_profiles = _parse_transfer_profiles(args.disagg_transfer_log)
    disagg = _build_disagg_analysis(disagg_metrics, transfer_profiles)

    payload = {
        "schema_version": 1,
        "metric_definitions": {
            "baseline_communication": "CUDA-event sum of explicitly wrapped collectives on one rank; includes collective wait/stall and is a subset of pipeline time.",
            "disagg_sender_wire_transfer": "Sender-side wall time spent inside synchronous Mooncake transfer_sync calls, summed across both inter-stage transfers; reported independently because sender and receiver scopes have different boundaries.",
            "disagg_critical_path_wire_transfer": "The sender transfer_sync duration capped by the corresponding receiver-visible interval, so only wire time observed on the controller critical path is allocated.",
            "disagg_transfer_backpressure": "Receiver-visible transfer-completion delay minus the allocated critical-path wire time; includes sender/sidecar queueing and status propagation.",
            "active_batch": "Maximum request completion relative to burst dispatch; teardown is excluded.",
        },
        "baseline_service": baseline_service,
        "baseline_ranked_profile": baseline_ranked,
        "disagg_profile": disagg,
        "comparison": {
            "throughput_speedup": disagg["throughput_per_min"] / baseline_service["throughput_per_min"],
            "baseline_communication_share_of_profiled_pipeline": baseline_ranked["communication_share_of_pipeline_critical_rank"],
            "disagg_sender_wire_share_of_mean_e2e": disagg["sender_wire_share_of_mean_e2e"],
            "disagg_critical_path_wire_share_of_mean_e2e": disagg["critical_path_wire_share_of_mean_e2e"],
            "disagg_transport_related_visible_share_of_mean_e2e": disagg["transport_related_visible_share_of_mean_e2e"],
        },
        "source_files": [
            str(args.baseline_metrics),
            str(args.baseline_ranked_log),
            str(args.disagg_metrics),
            str(args.disagg_transfer_log),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
