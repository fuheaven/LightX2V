#!/bin/bash

lightx2v_path=/home/fuhaiwen/LightX2V
model_path=/data/nvme1/models/qwen-image-edit-release-251130

export CUDA_VISIBLE_DEVICES=4

source ${lightx2v_path}/scripts/base/base.sh

# Start Encoder service after Transformer is ready
python -m lightx2v.server \
    --model_cls qwen_image \
    --task t2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/disagg/qwen/qwen_image_t2i_disagg_encoder.json \
    --host 0.0.0.0 \
    --port 8002

echo "Encoder service stopped"
