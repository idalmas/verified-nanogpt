# Backward-pass kernel benchmark: default vs deterministic

GPU: NVIDIA H100 PCIe · driver 580.159.04 · torch 2.8.0+cu128 · CUDA 12.8 · cuDNN 91002 · flash_attn not installed
Run: full · bf16 · TF32 off · 10 warm-up + 50 timed iters (median shown, p90 in results.json) · bit-identity over 5 backward runs

| Op | Impl | Shape | Default median ms | Det. median ms | Overhead (×) | Default bit-identical? | Det. bit-identical? |
|---|---|---|---|---|---|---|---|
| attention | flash_attn | B=1 H=64 S=8192 D=128 | N/A | N/A | — | — | — |
| attention | flash_attn | B=4 H=32 S=4096 D=128 | N/A | N/A | — | — | — |
| attention | flash_attn | B=8 H=16 S=2048 D=64 | N/A | N/A | — | — | — |
| attention | sdpa-auto (def→FlashAttention, det→FlashAttention) | B=1 H=64 S=8192 D=128 | 13.173 | 24.632 | 1.87 | **no** (5 distinct / 5) | yes (5/5) |
| attention | sdpa-flash (def→FlashAttention, det→FlashAttention) | B=1 H=64 S=8192 D=128 | 13.068 | 24.657 | 1.89 | **no** (5 distinct / 5) | yes (5/5) |
| attention | sdpa-efficient (def→EfficientAttention, det→EfficientAttention) | B=1 H=64 S=8192 D=128 | 29.005 | 41.986 | 1.45 | **no** (5 distinct / 5) | yes (5/5) |
| attention | sdpa-cudnn (def→CudnnAttention) | B=1 H=64 S=8192 D=128 | 8.437 | ERR | — | **no** (5 distinct / 5) | — |
| attention | sdpa-math (def→math, det→math) | B=1 H=64 S=8192 D=128 | 187.276 | 190.635 | 1.02 | yes (5/5) | yes (5/5) |
| attention | sdpa-auto (def→FlashAttention, det→FlashAttention) | B=4 H=32 S=4096 D=128 | 7.454 | 13.583 | 1.82 | **no** (5 distinct / 5) | yes (5/5) |
| attention | sdpa-flash (def→FlashAttention, det→FlashAttention) | B=4 H=32 S=4096 D=128 | 7.421 | 13.598 | 1.83 | **no** (5 distinct / 5) | yes (5/5) |
| attention | sdpa-efficient (def→EfficientAttention, det→EfficientAttention) | B=4 H=32 S=4096 D=128 | 15.013 | 23.270 | 1.55 | yes (5/5) | yes (5/5) |
| attention | sdpa-cudnn (def→CudnnAttention) | B=4 H=32 S=4096 D=128 | 4.368 | ERR | — | **no** (5 distinct / 5) | — |
| attention | sdpa-math (def→math, det→math) | B=4 H=32 S=4096 D=128 | 95.930 | 96.236 | 1.00 | yes (5/5) | yes (5/5) |
| attention | sdpa-auto (def→FlashAttention, det→FlashAttention) | B=8 H=16 S=2048 D=64 | 1.233 | 2.119 | 1.72 | **no** (5 distinct / 5) | yes (5/5) |
| attention | sdpa-flash (def→FlashAttention, det→FlashAttention) | B=8 H=16 S=2048 D=64 | 1.232 | 2.121 | 1.72 | **no** (5 distinct / 5) | yes (5/5) |
| attention | sdpa-efficient (def→EfficientAttention, det→EfficientAttention) | B=8 H=16 S=2048 D=64 | 2.527 | 3.877 | 1.53 | **no** (5 distinct / 5) | yes (5/5) |
| attention | sdpa-cudnn (def→CudnnAttention) | B=8 H=16 S=2048 D=64 | 0.797 | ERR | — | **no** (5 distinct / 5) | — |
| attention | sdpa-math (def→math, det→math) | B=8 H=16 S=2048 D=64 | 15.670 | 15.762 | 1.01 | yes (5/5) | yes (5/5) |
| embedding | nn.Embedding | V=128256 dim=8192 T=32768 | 4.177 | 5.821 | 1.39 | yes (5/5) | yes (5/5) |
| embedding | nn.Embedding | V=50304 dim=4096 T=65536 | 2.745 | 3.433 | 1.25 | yes (5/5) | yes (5/5) |
| matmul | nn.Linear | T=32768 in=8192 out=8192 | 19.508 | 19.484 | 1.00 | yes (5/5) | yes (5/5) |
| matmul | nn.Linear | T=32768 in=8192 out=28672 | 79.978 | 81.132 | 1.01 | yes (5/5) | yes (5/5) |
| matmul | nn.Linear | T=8192 in=4096 out=4096 | 1.140 | 1.139 | 1.00 | yes (5/5) | yes (5/5) |

## Failures / unsupported

- attention / flash_attn / B=1 H=64 S=8192 D=128 / default: **unavailable** (setup) — `flash_attn not installed: No module named 'flash_attn'`
- attention / flash_attn / B=1 H=64 S=8192 D=128 / deterministic: **unavailable** (setup) — `flash_attn not installed: No module named 'flash_attn'`
- attention / flash_attn / B=4 H=32 S=4096 D=128 / default: **unavailable** (setup) — `flash_attn not installed: No module named 'flash_attn'`
- attention / flash_attn / B=4 H=32 S=4096 D=128 / deterministic: **unavailable** (setup) — `flash_attn not installed: No module named 'flash_attn'`
- attention / flash_attn / B=8 H=16 S=2048 D=64 / default: **unavailable** (setup) — `flash_attn not installed: No module named 'flash_attn'`
- attention / flash_attn / B=8 H=16 S=2048 D=64 / deterministic: **unavailable** (setup) — `flash_attn not installed: No module named 'flash_attn'`
- attention / sdpa-cudnn / B=1 H=64 S=8192 D=128 / deterministic: **error** (forward) — `RuntimeError: No available kernel. Aborting execution. | warnings: cuDNN SDPA is not deterministic. (Triggered internally at /pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:604.); Memory efficient kernel not used because: (Triggered internally at /pytorch/aten/src/ATen/native/transform`
- attention / sdpa-cudnn / B=4 H=32 S=4096 D=128 / deterministic: **error** (forward) — `RuntimeError: No available kernel. Aborting execution. | warnings: cuDNN SDPA is not deterministic. (Triggered internally at /pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:604.); Memory efficient kernel not used because: (Triggered internally at /pytorch/aten/src/ATen/native/transform`
- attention / sdpa-cudnn / B=8 H=16 S=2048 D=64 / deterministic: **error** (forward) — `RuntimeError: No available kernel. Aborting execution. | warnings: cuDNN SDPA is not deterministic. (Triggered internally at /pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:604.); Memory efficient kernel not used because: (Triggered internally at /pytorch/aten/src/ATen/native/transform`

SDPA labels show the backward actually recorded by autograd (grad_fn) in each mode. Timing = backward only, via torch.autograd.grad on a retained graph (forward runs once, outside the timed region).
