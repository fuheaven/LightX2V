# wan2.1 i2v bf16 40 steps

## baseline

4090 24G
cpu_offload: block 峰值显存：9G     |   720P+block 峰值显存： 14G
DiT: 24.24 s/step                  |    90.71s/step
image encoder: 0.57s               |    0.52s
vae encoder: 3.02s                  |   6.83s
text encoder: 2.14s                 |   1.90s
total encoder: 6.09s                |   9.78s

## disagg

cpu_offload: block      |   720P+block
DiT: 19.02 s/step       |   DiT:  62.80s/step
image encoder: 0.28s    |   0.20s
vae encoder: 3.22s      |   7.36s
text encoder: 0.20s     |   0.20s
total encoder: 4.08s    |   8.32s

# wan2.1 i2v int8

## baseline + offload

use_offload: false会oom
cpu_offload: block
DiT: 12.94 s/step
image encoder: 0.79s
vae encoder: 3.11s
text encoder: 1.67s
total encoder: 5.90s

## disagg(no offload)

DiT: 12.55 s/step
image encoder: 0.19s
vae encoder: 2.93s
text encoder: 0.20s
total encoder: 3.69s

## disagg(offload)

DiT: 12.91s
image encoder: 0.19s
vae encoder: 2.93s
text encoder: 0.20s
total encoder: 3.70s

# concurrent benchmark template (4090)

> Run scripts:
>
> - baseline: `python scripts/server/4090/bench_concurrent_baseline.py --image_path <img> --concurrency_list 1,2,4,8`
> - disagg: `python scripts/server/4090/bench_concurrent_disagg.py --image_path <img> --concurrency <N>`
> - qwen t2i baseline: `python scripts/server/4090/bench_concurrent_qwen_baseline.py --concurrency_list 1,2,4,8`
> - qwen t2i disagg: `python scripts/server/4090/bench_concurrent_qwen_disagg.py --concurrency_list 1,2,4,8`
>
> Suggested N: 1 / 2 / 4 / 8, repeat 3 times, record median.

## BF16 + block offload


| N(concurrency) | mode     | ok/total | QPS    | P50(s) | P95(s) | P99(s) | note              |
| -------------- | -------- | -------- | ------ | ------ | ------ | ------ | ----------------- |
| 1              | baseline | 1        | 0.0091 | 109.45 | 109.45 | 109.45 | wan2.1 480P 4step |
| 2              | baseline | 2        | 0.0092 | 162.52 | 211.01 | 215.32 | wan2.1 480P 4step |
| 4              | baseline | 4        | 0.0093 | 269.85 | 414.94 | 427.83 | wan2.1 480P 4step |
| 8              | baseline | 8        | 0.0093 | 485.82 | 824.24 | 854.32 | wan2.1 480P 4step |
| 1              | disagg   | 1        | 0.0117 | 85.38  | 85.38  | 85.38  | wan2.1 480P 4step |
| 2              | disagg   | 2        | 0.0122 | 125.83 | 160.18 | 163.23 | wan2.1 480P 4step |
| 4              | disagg   | 4        | 0.0126 | 201.85 | 305.73 | 314.94 | wan2.1 480P 4step |
| 8              | disagg   | 8        | 0.0129 | 358.20 | 595.12 | 616.48 | wan2.1 480P 4step |



| N(concurrency) | mode     | ok/total | QPS    | P50(s) | P95(s) | P99(s) | note                  |
| -------------- | -------- | -------- | ------ | ------ | ------ | ------ | --------------------- |
| 1              | baseline | 1        | 0.0207 | 48.30  | 48.30  | 48.30  | qwen-image-2512 5step |
| 2              | baseline | 2        | 0.0207 | 72.57  | 94.31  | 94.24  | qwen-image-2512 5step |
| 4              | baseline | 4        | 0.0212 | 118.77 | 181.44 | 187.33 | qwen-image-2512 5step |
| 8              | baseline | 8        | 0.0216 | 208.64 | 354.94 | 367.90 | qwen-image-2512 5step |
| 1              | disagg   | 1        | 0.0452 | 22.11  | 22.11  | 22.11  | qwen-image-2512 5step |
| 2              | disagg   | 2        | 0.0510 | 29.68  | 38.25  | 39.02  | qwen-image-2512 5step |
| 4              | disagg   | 4        | 0.0528 | 48.52  | 73.00  | 75.17  | qwen-image-2512 5step |
| 8              | disagg   | 8        | 0.0534 | 85.78  | 143.37 | 148.62 | qwen-image-2512 5step |


