#!/usr/bin/env python3
"""
Start multiple Qwen Image disaggregated stacks on one host.

Each replica uses 2 GPUs by default (same layout as scripts/server/4090/start_qwen_t2i_disagg_8gpu_4replicas.sh):
  - ED GPU: Encoder + Decoder (same CUDA device index)
  - T  GPU: Transformer

Example (4 GPUs -> 2 replicas): ED on 0,2 and Transformer on 1,3
  LIGHTX2V_PATH=/path/to/LightX2V QWEN_IMAGE_MODEL_PATH=/path/to/model \\
    python start_multi_servers_disagg.py --task i2i --ed-gpus 0,2 --t-gpus 1,3

Per-replica HTTP ports (defaults for i2i: encoder 8012/8022/..., transformer 8013/8023/..., decoder 8014/8024/...).
Client traffic goes to the **decoder** port (same as single disagg).

Mooncake / P2P: each replica gets unique bootstrap_room / decoder_bootstrap_room via temp JSON configs.
ZMQ: set LIGHTX2V_DISAGG_PORT_OFFSET per replica (same value for enc/trans/dec of that replica); see lightx2v/disagg/conn.py.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _http_get_status(url: str, timeout_s: float = 2.0) -> str:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return str(getattr(resp, "status", 200))
    except urllib.error.HTTPError as e:
        return str(e.code)
    except Exception:
        return "000"


def wait_for_ready(port: int, pid: int | None, log_path: Path, timeout_s: int, interval_s: float, label: str) -> None:
    url = f"http://127.0.0.1:{port}/v1/service/status"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        code = _http_get_status(url)
        if code == "200":
            return
        if pid is not None:
            try:
                os.kill(pid, 0)
            except OSError:
                tail = ""
                if log_path.is_file():
                    try:
                        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        tail = "\n".join(lines[-80:])
                    except OSError:
                        tail = ""
                raise RuntimeError(
                    f"{label}: process exited while waiting for readiness (port={port}). Last log lines:\n{tail}"
                )
        time.sleep(interval_s)
    tail = ""
    if log_path.is_file():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-80:])
        except OSError:
            tail = ""
    raise TimeoutError(f"{label}: timeout waiting for {url}\nLast log lines:\n{tail}")


def patch_encoder(src: Path, dst: Path, phase1_room: int) -> None:
    data: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
    data["disagg_config"]["bootstrap_room"] = phase1_room
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def patch_transformer(src: Path, dst: Path, phase1_room: int, phase2_room: int) -> None:
    data: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
    data["disagg_config"]["bootstrap_room"] = phase1_room
    data["disagg_config"]["decoder_bootstrap_room"] = phase2_room
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def patch_decoder(src: Path, dst: Path, phase2_room: int) -> None:
    data: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
    data["disagg_config"]["bootstrap_room"] = phase2_room
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_int_list(s: str) -> list[int]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    return [int(x) for x in parts]


def main() -> None:
    ap = argparse.ArgumentParser(description="Start multiple Qwen Image disagg stacks (decoder URL is the client entry).")
    ap.add_argument("--task", type=str, choices=("i2i", "t2i"), default="i2i", help="Must match disagg config family.")
    ap.add_argument(
        "--lightx2v-path",
        type=str,
        default=os.environ.get("LIGHTX2V_PATH", ""),
        help="Repo root (or set LIGHTX2V_PATH).",
    )
    ap.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("QWEN_IMAGE_MODEL_PATH", ""),
        help="Model dir (or set QWEN_IMAGE_MODEL_PATH).",
    )
    ap.add_argument("--ed-gpus", type=str, required=True, help="Comma-separated ED GPUs (encoder+decoder each replica), e.g. 0,2")
    ap.add_argument("--t-gpus", type=str, required=True, help="Comma-separated Transformer GPUs, e.g. 1,3")
    ap.add_argument("--port-step", type=int, default=10, help="Port increment between replicas.")
    ap.add_argument("--enc-port-base", type=int, default=None, help="Encoder port for replica 0 (default: i2i 8012, t2i 8002).")
    ap.add_argument("--trans-port-base", type=int, default=None, help="Transformer port for replica 0 (default: i2i 8013, t2i 8003).")
    ap.add_argument("--dec-port-base", type=int, default=None, help="Decoder port for replica 0 (default: i2i 8014, t2i 8008).")
    ap.add_argument("--base-phase1-room", type=int, default=100, help="Phase1 bootstrap_room for replica 0 (+idx per replica).")
    ap.add_argument("--base-phase2-room", type=int, default=200, help="Phase2 bootstrap_room for replica 0 (+idx per replica).")
    ap.add_argument("--tmp-dir", type=str, default="/tmp/qwen_disagg_cfg_multi", help="Temp JSON configs per replica.")
    ap.add_argument("--log-dir", type=str, default="/tmp/qwen_disagg_logs_multi", help="Per-process stdout/stderr logs.")
    ap.add_argument("--startup-timeout-s", type=int, default=600, help="Max wait per service for /v1/service/status 200.")
    ap.add_argument("--ready-poll-interval-s", type=float, default=2.0, help="Polling interval for readiness.")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="Server bind address.")
    ap.add_argument(
        "--zmq-port-step",
        type=int,
        default=100,
        help="Per-replica ZMQ port offset increment (LIGHTX2V_DISAGG_PORT_OFFSET = replica_index * step).",
    )
    args = ap.parse_args()

    lightx2v_path = Path(args.lightx2v_path).resolve()
    if not lightx2v_path.is_dir():
        print("[ERROR] --lightx2v-path must be an existing directory.", file=sys.stderr)
        sys.exit(1)
    model_path = args.model_path.strip()
    if not model_path:
        print("[ERROR] --model-path or QWEN_IMAGE_MODEL_PATH is required.", file=sys.stderr)
        sys.exit(1)

    ed_gpus = parse_int_list(args.ed_gpus)
    t_gpus = parse_int_list(args.t_gpus)
    if len(ed_gpus) != len(t_gpus):
        print("[ERROR] --ed-gpus and --t-gpus must have the same length (one replica per pair).", file=sys.stderr)
        sys.exit(1)
    if not ed_gpus:
        print("[ERROR] empty GPU lists.", file=sys.stderr)
        sys.exit(1)

    if args.task == "i2i":
        enc_def, trans_def, dec_def = 8012, 8013, 8014
    else:
        enc_def, trans_def, dec_def = 8002, 8003, 8008

    enc_base = args.enc_port_base if args.enc_port_base is not None else enc_def
    trans_base = args.trans_port_base if args.trans_port_base is not None else trans_def
    dec_base = args.dec_port_base if args.dec_port_base is not None else dec_def

    cfg_dir = lightx2v_path / "configs" / "disagg" / "qwen"
    if args.task == "i2i":
        enc_src = cfg_dir / "qwen_image_i2i_disagg_encoder.json"
        trans_src = cfg_dir / "qwen_image_i2i_disagg_transformer.json"
        dec_src = cfg_dir / "qwen_image_i2i_disagg_decode.json"
    else:
        enc_src = cfg_dir / "qwen_image_t2i_disagg_encoder.json"
        trans_src = cfg_dir / "qwen_image_t2i_disagg_transformer.json"
        dec_src = cfg_dir / "qwen_image_t2i_disagg_decode.json"

    for p in (enc_src, trans_src, dec_src):
        if not p.is_file():
            print(f"[ERROR] Missing config template: {p}", file=sys.stderr)
            sys.exit(1)

    tmp_dir = Path(args.tmp_dir)
    log_dir = Path(args.log_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_sh = lightx2v_path / "scripts" / "base" / "base.sh"
    if not base_sh.is_file():
        print(f"[WARN] base.sh not found at {base_sh} (env may be incomplete).", file=sys.stderr)

    procs: list[subprocess.Popen[str]] = []

    def stop_all() -> None:
        print("\nStopping all disagg processes...")
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        print("Stopped.")

    def on_signal(_signum: int, _frame: Any) -> None:
        stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    decoder_urls: list[str] = []

    print(
        f"Starting {len(ed_gpus)} disagg replica(s) task={args.task} model_path={model_path}\n"
        f"tmp_dir={tmp_dir} log_dir={log_dir}"
    )

    try:
        _start_replicas(
            args,
            lightx2v_path,
            model_path,
            ed_gpus,
            t_gpus,
            enc_base,
            trans_base,
            dec_base,
            enc_src,
            trans_src,
            dec_src,
            tmp_dir,
            log_dir,
            procs,
            decoder_urls,
        )
    except Exception:
        stop_all()
        raise

    print("\nAll replicas are up. Client (bench) should use **decoder** base URL(s):")
    for r, u in enumerate(decoder_urls):
        print(f"  [{r}] {u}")
    print("\nBench example (I2I):")
    print(f"  python bench_qwen_disagg.py --url_list {','.join(decoder_urls)} --endpoint /v1/tasks/image/")
    print("\nPress Ctrl+C to stop.")
    try:
        while True:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    print(f"[ERROR] A server process exited with code {ret}. Stopping all.", file=sys.stderr)
                    stop_all()
                    sys.exit(1)
            time.sleep(2.0)
    except KeyboardInterrupt:
        stop_all()
        sys.exit(0)


def _start_replicas(
    args: argparse.Namespace,
    lightx2v_path: Path,
    model_path: str,
    ed_gpus: list[int],
    t_gpus: list[int],
    enc_base: int,
    trans_base: int,
    dec_base: int,
    enc_src: Path,
    trans_src: Path,
    dec_src: Path,
    tmp_dir: Path,
    log_dir: Path,
    procs: list[subprocess.Popen[str]],
    decoder_urls: list[str],
) -> None:
    for r, (ed_gpu, t_gpu) in enumerate(zip(ed_gpus, t_gpus)):
        phase1_room = args.base_phase1_room + r
        phase2_room = args.base_phase2_room + r
        enc_port = enc_base + r * args.port_step
        trans_port = trans_base + r * args.port_step
        dec_port = dec_base + r * args.port_step

        cfg_enc = tmp_dir / f"encoder_r{r}_{args.task}.json"
        cfg_trans = tmp_dir / f"transformer_r{r}_{args.task}.json"
        cfg_dec = tmp_dir / f"decoder_r{r}_{args.task}.json"

        patch_encoder(enc_src, cfg_enc, phase1_room)
        patch_transformer(trans_src, cfg_trans, phase1_room, phase2_room)
        patch_decoder(dec_src, cfg_dec, phase2_room)

        env = os.environ.copy()
        env.setdefault("lightx2v_path", str(lightx2v_path))
        env.setdefault("model_path", model_path)

        common_cmd_tail = [
            "-m",
            "lightx2v.server",
            "--model_cls",
            "qwen_image",
            "--task",
            args.task,
            "--model_path",
            model_path,
            "--host",
            args.host,
        ]

        zmq_off = str(r * args.zmq_port_step)
        print(
            f"\n[Replica {r}] ED_GPU={ed_gpu} T_GPU={t_gpu} "
            f"phase1_room={phase1_room} phase2_room={phase2_room} "
            f"ports enc/trans/dec={enc_port}/{trans_port}/{dec_port} "
            f"LIGHTX2V_DISAGG_PORT_OFFSET={zmq_off}"
        )

        # 1) Decoder first
        log_dec = log_dir / f"r{r}_decoder_{dec_port}.log"
        fdec = open(log_dec, "a", encoding="utf-8")
        env_d = env.copy()
        env_d["LIGHTX2V_DISAGG_PORT_OFFSET"] = zmq_off
        env_d["CUDA_VISIBLE_DEVICES"] = str(ed_gpu)
        cmd_d = [sys.executable, *common_cmd_tail, "--config_json", str(cfg_dec), "--port", str(dec_port)]
        print(f"  [1/3] Decoder  -> GPU {ed_gpu} log={log_dec}")
        p_dec = subprocess.Popen(cmd_d, env=env_d, stdout=fdec, stderr=subprocess.STDOUT, cwd=str(lightx2v_path))
        procs.append(p_dec)
        wait_for_ready(
            dec_port,
            p_dec.pid,
            log_dec,
            args.startup_timeout_s,
            args.ready_poll_interval_s,
            f"replica{r} decoder",
        )

        # 2) Transformer
        log_tr = log_dir / f"r{r}_transformer_{trans_port}.log"
        ftr = open(log_tr, "a", encoding="utf-8")
        env_t = env.copy()
        env_t["LIGHTX2V_DISAGG_PORT_OFFSET"] = zmq_off
        env_t["CUDA_VISIBLE_DEVICES"] = str(t_gpu)
        cmd_t = [sys.executable, *common_cmd_tail, "--config_json", str(cfg_trans), "--port", str(trans_port)]
        print(f"  [2/3] Transformer -> GPU {t_gpu} log={log_tr}")
        p_tr = subprocess.Popen(cmd_t, env=env_t, stdout=ftr, stderr=subprocess.STDOUT, cwd=str(lightx2v_path))
        procs.append(p_tr)
        wait_for_ready(
            trans_port,
            p_tr.pid,
            log_tr,
            args.startup_timeout_s,
            args.ready_poll_interval_s,
            f"replica{r} transformer",
        )

        # 3) Encoder
        log_enc = log_dir / f"r{r}_encoder_{enc_port}.log"
        fenc = open(log_enc, "a", encoding="utf-8")
        env_e = env.copy()
        env_e["LIGHTX2V_DISAGG_PORT_OFFSET"] = zmq_off
        env_e["CUDA_VISIBLE_DEVICES"] = str(ed_gpu)
        cmd_e = [sys.executable, *common_cmd_tail, "--config_json", str(cfg_enc), "--port", str(enc_port)]
        print(f"  [3/3] Encoder -> GPU {ed_gpu} log={log_enc}")
        p_enc = subprocess.Popen(cmd_e, env=env_e, stdout=fenc, stderr=subprocess.STDOUT, cwd=str(lightx2v_path))
        procs.append(p_enc)
        wait_for_ready(
            enc_port,
            p_enc.pid,
            log_enc,
            args.startup_timeout_s,
            args.ready_poll_interval_s,
            f"replica{r} encoder",
        )

        decoder_urls.append(f"http://127.0.0.1:{dec_port}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
