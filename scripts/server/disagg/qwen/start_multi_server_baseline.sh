#!/bin/bash
#
# Start Qwen Image T2I baseline with N independent server processes (1 GPU each).
#
# This script is designed to pair with:
#   - scripts/server/4090/bench_concurrent_qwen_baseline.py
#
# Default allocation:
#   GPU_LIST_CSV defaults to 0..7
#   Ports : 8200..(8200+N-1)
#
set -e
LIGHTX2V_PATH=/data/fuhaiwen1/LightX2V
MODEL_PATH=/workspace/Qwen-Image-2512/

# Model path (must be provided by your environment if different)
CONFIG_JSON="${LIGHTX2V_PATH}/configs/qwen_image/4090/qwen_image_t2i.json"

GPU_LIST_CSV="${GPU_LIST_CSV:-0,1,2,3}"
PORT_BASE="${PORT_BASE:-8200}"


export lightx2v_path="${LIGHTX2V_PATH}"
export model_path="${MODEL_PATH}"
source "${lightx2v_path}/scripts/base/base.sh"

IFS=',' read -r -a GPU_LIST <<< "${GPU_LIST_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "[ERROR] GPU_LIST_CSV is empty: ${GPU_LIST_CSV}"
  exit 1
fi
NUM_SERVERS="${#GPU_LIST[@]}"

PIDS=()
LOG_DIR="${LOG_DIR:-/tmp/qwen_baseline_logs}"
mkdir -p "${LOG_DIR}"
cleanup() {
  echo "Stopping all baseline servers..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "Stopped."
}
trap cleanup EXIT INT TERM

for i in $(seq 0 $((NUM_SERVERS-1))); do
  gpu=${GPU_LIST[$i]}
  port=$((PORT_BASE + i))
  echo "[${i}/${NUM_SERVERS}] Starting baseline server on GPU=${gpu}, port=${port} ..."
  CUDA_VISIBLE_DEVICES="${gpu}" python -m lightx2v.server \
    --model_cls qwen_image \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${CONFIG_JSON}" \
    --host 0.0.0.0 \
    --port "${port}" > "${LOG_DIR}/baseline_${port}.log" 2>&1 &
  pid=$!
  PIDS+=("${pid}")

  echo "[${i}/${NUM_SERVERS}] Waiting for readiness: port=${port} ..."
  sleep 30
done

echo ""
echo "Baseline servers started. URLs:"
for i in $(seq 0 $((NUM_SERVERS-1))); do
  port=$((PORT_BASE + i))
  echo "  - http://localhost:${port}"
done
echo ""
echo "Press Ctrl+C to stop."
wait