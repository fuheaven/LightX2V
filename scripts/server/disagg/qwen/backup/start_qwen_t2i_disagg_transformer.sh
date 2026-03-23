#!/bin/bash

lightx2v_path=/home/fuhaiwen/LightX2V
model_path=/data/nvme1/models/qwen-image-edit-release-251130

export CUDA_VISIBLE_DEVICES=5

source ${lightx2v_path}/scripts/base/base.sh

# Start Transformer service first, it will wait for Encoder to connect via Mooncake
python -m lightx2v.server \
    --model_cls qwen_image \
    --task t2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/disagg/qwen/qwen_image_t2i_disagg_transformer.json \
    --host 0.0.0.0 \
    --port 8003

echo "Transformer service stopped"
