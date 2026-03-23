#!/bin/bash
#
# Start Qwen Image T2I disagg with shared ED + multi-transformer on one host.
#
# Mirrors:
#   - start_multi_server_baseline.sh  : CSV GPU lists, per-replica loop, LOG_DIR, PIDS cleanup
#   - start_qwen_t2i_disagg.sh      : Decoder -> Transformer -> Encoder, t2i disagg JSONs, ports 8002/8003/8008 (replica 0)
#
# Topology:
#   - One shared ED pair: Encoder + Decoder on ED_GPU
#   - Multiple Transformer services on T_GPUS_CSV (default 3)
#
# Example:
#   ED_GPU=0 T_GPUS_CSV=1,2,3 ./start_multi_server_disagg.sh
#
# Mooncake: shared ED + multi-transformers all use same phase rooms.
# ZMQ (lightx2v/disagg/conn.py): every process on same host must use unique
# LIGHTX2V_DISAGG_PORT_OFFSET to avoid bind conflicts.
# Client: official 3-way flow is POST Decoder -> Transformer -> Encoder (same JSON), then poll Decoder.
# bench_qwen_disagg.py does this by default; do not only hit Decoder HTTP or Encoder/Transformer will idle.
#
# Startup: no HTTP readiness checks. Launch Decoder + Encoder first, then launch
# each Transformer with REPLICA_STAGGER_SECS spacing.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTX2V_PATH="${LIGHTX2V_PATH:-/data/fuhaiwen1/LightX2V}"
model_path="${QWEN_IMAGE_MODEL_PATH:-/workspace/Qwen-Image-2512/}"

ED_GPU="${ED_GPU:-0}"
T_GPUS_CSV="${T_GPUS_CSV:-1,2,3}"

PORT_STEP="${PORT_STEP:-10}"
ENC_PORT_BASE="${ENC_PORT_BASE:-8002}"
TRANS_PORT_BASE="${TRANS_PORT_BASE:-8003}"
DEC_PORT_BASE="${DEC_PORT_BASE:-8008}"

PHASE1_ROOM="${PHASE1_ROOM:-1000000}"
PHASE2_ROOM="${PHASE2_ROOM:-2000000}"
ROOM_STRIDE="${ROOM_STRIDE:-100000}"

TMP_DIR="${TMP_DIR:-/tmp/qwen_t2i_disagg_cfg_multi}"
LOG_DIR="${LOG_DIR:-/tmp/qwen_t2i_disagg_logs_multi}"
# Seconds to sleep after launching one full 3-way stack before starting the next replica
REPLICA_STAGGER_SECS="${REPLICA_STAGGER_SECS:-30}"
# Per-process ZMQ port offset increment (see LIGHTX2V_DISAGG_PORT_OFFSET in lightx2v/disagg/conn.py)
ZMQ_PORT_STEP="${ZMQ_PORT_STEP:-100}"

export lightx2v_path="${LIGHTX2V_PATH}"
export model_path="${model_path}"
source "${lightx2v_path}/scripts/base/base.sh"

IFS=',' read -r -a T_GPUS <<< "${T_GPUS_CSV}"

NUM_TRANSFORMERS="${#T_GPUS[@]}"
if [ "${NUM_TRANSFORMERS}" -lt 1 ]; then
  echo "[ERROR] Empty T_GPUS_CSV."
  exit 1
fi

CONFIG_BASE="${LIGHTX2V_PATH}/configs/disagg/qwen"
ENC_SRC="${CONFIG_BASE}/qwen_image_t2i_disagg_encoder.json"
TRANS_SRC="${CONFIG_BASE}/qwen_image_t2i_disagg_transformer.json"
DEC_SRC="${CONFIG_BASE}/qwen_image_t2i_disagg_decode.json"

for f in "${ENC_SRC}" "${TRANS_SRC}" "${DEC_SRC}"; do
  if [ ! -f "${f}" ]; then
    echo "[ERROR] Missing config: ${f}"
    exit 1
  fi
done

mkdir -p "${TMP_DIR}" "${LOG_DIR}"

PIDS=()
cleanup() {
  echo "Stopping all Qwen T2I multi disagg services..."
  for pid in "${PIDS[@]}"; do
    [ -n "${pid}" ] && kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "Stopped."
}
trap cleanup EXIT INT TERM

echo "Starting shared-ED + ${NUM_TRANSFORMERS} Transformer(s); tmp=${TMP_DIR} logs=${LOG_DIR} stagger=${REPLICA_STAGGER_SECS}s"

enc_port="${ENC_PORT_BASE}"
dec_port="${DEC_PORT_BASE}"
cfg_enc="${TMP_DIR}/encoder_shared.json"
cfg_dec="${TMP_DIR}/decoder_shared.json"

python - "${ENC_SRC}" "${cfg_enc}" "${PHASE1_ROOM}" "${ROOM_STRIDE}" "${NUM_TRANSFORMERS}" <<'PY'
import json, sys
src, dst, base1, stride, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rooms = [base1 + i * stride for i in range(n)]
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)
# Pre-initialize multiple bootstrap rooms for shared encoder.
data["disagg_config"]["bootstrap_rooms"] = rooms
# Keep bootstrap_room as the first element for backward compatibility.
data["disagg_config"]["bootstrap_room"] = rooms[0]
with open(dst, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PY

python - "${DEC_SRC}" "${cfg_dec}" "${PHASE2_ROOM}" "${ROOM_STRIDE}" "${NUM_TRANSFORMERS}" <<'PY'
import json, sys
src, dst, base2, stride, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rooms2 = [base2 + i * stride for i in range(n)]
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)
# Pre-initialize multiple bootstrap rooms for shared decoder (phase2).
data["disagg_config"]["bootstrap_rooms"] = rooms2
data["disagg_config"]["bootstrap_room"] = rooms2[0]
with open(dst, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PY

# Shared ZMQ offset for all services in this "shared ED + multi-transformer" topology.
# Do NOT use unique offset per transformer; Encoder->Transformer phase1 status and
# Transformer->Decoder phase2 transfers must land on the same ports.
SHARED_ZMQ_PORT_OFFSET="${SHARED_ZMQ_PORT_OFFSET:-0}"

echo ""
echo "[Shared ED] ED_GPU=${ED_GPU} phase1_room=${PHASE1_ROOM} phase2_room=${PHASE2_ROOM}"
echo "  ports: encoder=${enc_port} decoder=${dec_port}"
echo "  LIGHTX2V_DISAGG_PORT_OFFSET: shared=${SHARED_ZMQ_PORT_OFFSET}"
echo "  Launching Decoder + Encoder (background)..."

log_dec="${LOG_DIR}/shared_decoder_${dec_port}.log"
LIGHTX2V_DISAGG_PORT_OFFSET="${SHARED_ZMQ_PORT_OFFSET}" CUDA_VISIBLE_DEVICES="${ED_GPU}" python -m lightx2v.server \
  --model_cls qwen_image \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${cfg_dec}" \
  --host 0.0.0.0 \
  --port "${dec_port}" > "${log_dec}" 2>&1 &
PIDS+=($!)

log_enc="${LOG_DIR}/shared_encoder_${enc_port}.log"
LIGHTX2V_DISAGG_PORT_OFFSET="${SHARED_ZMQ_PORT_OFFSET}" CUDA_VISIBLE_DEVICES="${ED_GPU}" python -m lightx2v.server \
  --model_cls qwen_image \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${cfg_enc}" \
  --host 0.0.0.0 \
  --port "${enc_port}" > "${log_enc}" 2>&1 &
PIDS+=($!)

for r in $(seq 0 $((NUM_TRANSFORMERS - 1))); do
  t_gpu="${T_GPUS[$r]}"
  trans_port=$((TRANS_PORT_BASE + r * PORT_STEP))
  cfg_trans="${TMP_DIR}/transformer_r${r}.json"
  phase1_room=$((PHASE1_ROOM + r * ROOM_STRIDE))
  phase2_room=$((PHASE2_ROOM + r * ROOM_STRIDE))

  python - "${TRANS_SRC}" "${cfg_trans}" "${phase1_room}" "${phase2_room}" "${r}" <<'PY'
import json, sys
src, dst, room1, room2, r = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)
data["disagg_config"]["bootstrap_room"] = room1
data["disagg_config"]["decoder_bootstrap_room"] = room2

# IMPORTANT:
# For transformer phase1, DataManager binds:
#   tcp://*: (DATARECEIVER_POLLING_PORT + receiver_engine_rank)
# If multiple transformers share the same receiver_engine_rank,
# they will crash with "Address already in use".
# Therefore assign a unique receiver_engine_rank per transformer process.
data["disagg_config"]["receiver_engine_rank"] = 100 + r
with open(dst, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PY

  echo ""
  echo "[Transformer ${r}/${NUM_TRANSFORMERS}] T_GPU=${t_gpu}"
  echo "  port: transformer=${trans_port}"
  echo "  LIGHTX2V_DISAGG_PORT_OFFSET=${SHARED_ZMQ_PORT_OFFSET} (shared)"
  echo "  Launching Transformer (background)..."

  log_tr="${LOG_DIR}/r${r}_transformer_${trans_port}.log"
  LIGHTX2V_DISAGG_PORT_OFFSET="${SHARED_ZMQ_PORT_OFFSET}" CUDA_VISIBLE_DEVICES="${t_gpu}" python -m lightx2v.server \
    --model_cls qwen_image \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${cfg_trans}" \
    --host 0.0.0.0 \
    --port "${trans_port}" > "${log_tr}" 2>&1 &
  PIDS+=($!)

  if [ "${r}" -lt $((NUM_TRANSFORMERS - 1)) ]; then
    echo "  Waiting ${REPLICA_STAGGER_SECS}s before starting next transformer..."
    sleep "${REPLICA_STAGGER_SECS}"
  fi
done

echo ""
echo "All services launched (no readiness wait). Bench/client URLs:"
echo "  decoder: http://localhost:${dec_port}"
echo "  encoder: http://localhost:${enc_port}"
echo "  transformers:"
for r in $(seq 0 $((NUM_TRANSFORMERS - 1))); do
  trans_port=$((TRANS_PORT_BASE + r * PORT_STEP))
  echo "    [${r}] http://localhost:${trans_port}"
done
echo ""
echo "Press Ctrl+C to stop."
wait
