# Wan VAE 优化技术评估报告

> [!CAUTION]
> **实测结论：TensorRT 方案不可行，CUDA Graphs 收益有限**。详见下文分析。

---

## 执行摘要

| 优化方案 | 状态 | 加速效果 |
| :--- | :--- | :--- |
| **TensorRT** | ❌ 阻塞 | 多尺度 Cache 与 ONNX Dynamic Axes 不兼容 |
| **CUDA Graphs** | ✅ 可用 | **~9%** (94.67ms → 86.74ms) |

**建议**：保持 PyTorch 原生实现，优化重心放在 DiT Transformer。

---

## 1. TensorRT 方案 (已放弃)

### 1.1 技术障碍

Wan VAE 采用 **Streaming (流式)** 架构，每帧解码需维护 30+ 个 Cache Tensor。这些 Cache 分布在不同网络层，**空间尺度各不相同**：

| 层级 | Cache 尺寸示例 (480p) |
| :--- | :--- |
| 第 1 层 | `[1, 128, 2, 60, 104]` |
| 下采样后 | `[1, 256, 2, 30, 52]` |
| 最深层 | `[1, 512, 2, 15, 26]` |

ONNX 的 Dynamic Axes 假设所有动态维度成比例变化，但 Cache 的 H/W 是**层依赖的**，无法统一描述。

### 1.2 实测错误

```
RuntimeError: Sizes of tensors must match except in dimension 2. 
Expected size 512 but got size 4.
```

**结论**：需完全重构模型架构才能适配 TRT，工程成本过高。

---

## 2. CUDA Graphs 方案 (可选)

CUDA Graphs 通过捕获 Kernel 执行序列并重放，消除 Kernel Launch Overhead。

### 2.1 测试结果

**简单模型 (Conv3d)**:
```
Baseline:    0.044 ms
CUDA Graph:  0.014 ms
Speedup:     3.09x
```

**Decoder3d (128M 参数)**:
```
Baseline:    94.67 ms
CUDA Graph:  86.74 ms
Speedup:     1.09x
```

### 2.2 分析

CUDA Graphs 对计算密集型模型（如 Decoder3d）加速有限：
- **Kernel Launch Overhead** 在 94ms 计算中占比 < 10%
- 主要收益来自消除 Python 调度开销

### 2.3 使用方式 (可选)

如仍需启用，可使用封装器：

```python
from lightx2v.models.video_encoders.cuda_graphs.wan_vae_cuda_graph import wrap_vae_with_cuda_graphs

vae = WanVAE(...)
vae_with_graph = wrap_vae_with_cuda_graphs(vae, warmup_runs=3)
output = vae_with_graph.decode(z, scale)
```

---

## 3. 建议

1. **VAE 优先级降低**：VAE 在整体 Pipeline 中耗时占比 ~10%，优化 DiT 收益更大。
2. **保持原生 PyTorch**：对计算密集型组件，PyTorch + Compile 是更实际的选择。
3. **关注 DiT 优化**：参考 `examples/BeginnerGuide/ZH_CN/DiTTensorRTOptimize.md`（如有）。

---

## 附录：相关文件

| 文件 | 用途 |
| :--- | :--- |
| `tools/convert/tensorrt/wan/convert_wan_vae_trt.py` | TRT 转换脚本 (仅供参考) |
| `lightx2v/models/video_encoders/cuda_graphs/wan_vae_cuda_graph.py` | CUDA Graphs 封装器 |
| `tools/benchmark/benchmark_decoder_step.py` | 性能基准测试脚本 |
