#!/bin/bash

lightx2v_path=
model_path=/data/nvme0/models/Wan-AI/Wan2.1-T2V-14B

export CUDA_VISIBLE_DEVICES=

source ${lightx2v_path}/scripts/base/base.sh

# Start Encoder service after Transformer is ready
python -m lightx2v.server \
    --model_cls wan2.1 \
    --task t2v \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/disagg/wan/wan_t2v_disagg_encoder.json \
    --host 0.0.0.0 \
    --port 8002

echo "Encoder service stopped"
