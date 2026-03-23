# Qwen Image T2I / I2I 三段式分离部署

与 Wan 一致的三段式流程：**先请求 Decoder → 再请求 Transformer → 最后请求 Encoder**，结果在 Decoder 节点保存。

## T2I 3-way

### 启动服务（顺序：Decoder → Transformer → Encoder）

```bash
cd /home/fuhaiwen/LightX2V
# 可选：export QWEN_IMAGE_MODEL_PATH=/path/to/qwen-image-edit-release-251130
# 可选：GPU_ENCODER=4 GPU_TRANSFORMER=5 GPU_DECODER=4
bash scripts/server/disagg/qwen/start_qwen_t2i_disagg_3way.sh
```

- Encoder: port 8002  
- Transformer: port 8003  
- Decoder: port 8008  

### 发请求（客户端）

```bash
cd /home/fuhaiwen/LightX2V
python scripts/server/disagg/qwen/post_qwen_t2i_3way.py
```

### 一键检查服务并跑一次请求

```bash
bash scripts/server/disagg/qwen/test_qwen_t2i_3way.sh
```

输出图片在 **Decoder 进程所在机器** 的当前工作目录下的 `save_results/qwen_t2i_disagg_3way.png`。

---

## I2I 3-way

### 启动服务

```bash
bash scripts/server/disagg/qwen/start_qwen_i2i_disagg_3way.sh
```

- Encoder: port 8012  
- Transformer: port 8013  
- Decoder: port 8014  

### 发请求

修改 `post_qwen_i2i_3way.py` 中的 `IMAGE_PATH` 为本地 I2I 输入图路径，然后：

```bash
python scripts/server/disagg/qwen/post_qwen_i2i_3way.py
```

### 一键测试

```bash
bash scripts/server/disagg/qwen/test_qwen_i2i_3way.sh
```

结果在 Decoder 节点的 `save_results/qwen_i2i_disagg_3way.png`。

---

## 配置文件

- T2I: `configs/disagg/qwen/qwen_image_t2i_disagg_encoder.json` / `qwen_image_t2i_disagg_transformer.json` / `qwen_image_t2i_disagg_decode.json`
- I2I: `configs/disagg/qwen/qwen_image_i2i_disagg_encoder.json` / `qwen_image_i2i_disagg_transformer.json` / `qwen_image_i2i_disagg_decode.json`

Transformer 配置中已包含 `decoder_engine_rank` 与 `decoder_bootstrap_room`，用于 Phase2 向 Decoder 发送 latents。
