# Backward-pass kernel benchmark

`kernel_bench.py` measures what bitwise-deterministic training costs at frontier-scale tensor shapes.
It runs each key backward-pass op on its own, so kernel-launch overhead in a small model can't hide the cost.
Each op runs twice, once with default (possibly nondeterministic) kernels and once with deterministic ones. The script also checks whether repeated backward passes are bitwise identical.
It's standalone and imports nothing from the rest of the repo.

```bash
python benchmarks/kernel_bench.py           # full shapes, target: 1x H100 80GB, bf16
python benchmarks/kernel_bench.py --smoke   # tiny shapes, any CUDA GPU, about a minute
```

Useful flags: `--ops attention,embedding,matmul`, `--iters 50`, `--warmup 10`, `--bit-runs 5`, `--out-dir DIR`.

Outputs, written next to the script by default:
- `results.md`: the summary table, which is also printed
- `results.json`: raw per-iteration timings, p90, peak memory, gradient hashes and full error messages
- `env.json`: GPU, driver (from `nvidia-smi`), and the torch / CUDA / cuDNN / flash_attn versions

## Results (H100, 2026-10-10)

1× H100 PCIe 80GB, driver 580.159.04, torch 2.8.0+cu128, CUDA 12.8, cuDNN 9.10.2, bf16, TF32 off. Median backward time over 50 timed iters. Full table: [`results.md`](results.md).

| Op | Shape | Default ms | Det. ms | Overhead | Default bit-identical? |
|---|---|---|---|---|---|
| SDPA flash | B=1 H=64 S=8192 D=128 | 13.07 | 24.66 | 1.89× | no |
| SDPA flash | B=4 H=32 S=4096 D=128 | 7.42 | 13.60 | 1.83× | no |
| SDPA flash | B=8 H=16 S=2048 D=64 | 1.23 | 2.12 | 1.72× | no |
| SDPA efficient | B=1 H=64 S=8192 D=128 | 29.01 | 41.99 | 1.45× | no |
| SDPA efficient | B=4 H=32 S=4096 D=128 | 15.01 | 23.27 | 1.55× | yes |
| SDPA efficient | B=8 H=16 S=2048 D=64 | 2.53 | 3.88 | 1.53× | no |
| SDPA math | B=1 H=64 S=8192 D=128 | 187.28 | 190.64 | 1.02× | yes |
| SDPA math | B=4 H=32 S=4096 D=128 | 95.93 | 96.24 | 1.00× | yes |
| SDPA math | B=8 H=16 S=2048 D=64 | 15.67 | 15.76 | 1.01× | yes |
| SDPA cuDNN | all 3 shapes | 0.80–8.44 | ERR | — | no |
| `nn.Embedding` | V=128256 dim=8192 T=32768 | 4.18 | 5.82 | 1.39× | yes |
| `nn.Embedding` | V=50304 dim=4096 T=65536 | 2.75 | 3.43 | 1.25× | yes |
| `nn.Linear` | T=32768 in=8192 out=8192 | 19.51 | 19.48 | 1.00× | yes |
| `nn.Linear` | T=32768 in=8192 out=28672 | 79.98 | 81.13 | 1.01× | yes |
| `nn.Linear` | T=8192 in=4096 out=4096 | 1.14 | 1.14 | 1.00× | yes |

Every kernel that ran in deterministic mode was bit-identical across all 5 runs.

What stands out:
- **Attention is where determinism costs.** Deterministic flash backward, the fastest kernel that runs in deterministic mode and the one `sdpa-auto` picks, costs 1.72–1.89×. Memory-efficient costs 1.45–1.55× but starts about 2× slower than flash. Math is free to make deterministic but is 13–14× slower than flash.
- **cuDNN attention, the fastest default (0.80–8.44 ms), refuses deterministic mode.** It fails with `No available kernel`, so a deterministic run loses it entirely.
- **Embedding was already bit-identical in default mode**, even with Zipf-distributed token IDs chosen to cause contention. Deterministic mode still costs 1.25–1.39×.
- **Matmul is free (1.00–1.01×)**, as expected for the control.
- **`flash_attn` (FlashAttention-2) is untested** because it wasn't installed on the pod.

These are kernel-only numbers. End-to-end training overhead isn't measured yet. Deterministic mode in `train.py` also turns off `torch.compile`, which these numbers leave out.

## What each op tests

| Op | Impls | Why |
|---|---|---|
| Attention backward (causal, bf16) | `flash_attn` (`deterministic=False` vs `True`) when installed. SDPA with backend `auto`, `flash`, `efficient`, `cudnn` and `math`. | FlashAttention-2's backward accumulates dQ with atomics, so the default may be nondeterministic. For SDPA, the table shows which backward autograd actually recorded in each mode. A backend that raises under deterministic mode is recorded as `ERR`, with the message. |
| Embedding backward | `nn.Embedding`, with token IDs sampled from a Zipf(s=1.0) distribution over the vocab | Common tokens repeat many times in the batch. Those repeats cause atomic contention and scatter-add nondeterminism in the weight gradient. |
| Matmul backward (control) | `nn.Linear` without bias; gradients for both input and weight | cuBLAS GEMMs should cost about the same in both modes. This row is the baseline. |

## Method

- **Mode isolation:** each (op group, mode) pair runs in a separate subprocess. Deterministic mode sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the process starts, then calls `torch.use_deterministic_algorithms(True)` (not `warn_only`) and sets `cudnn.deterministic=True`. Default mode removes `CUBLAS_WORKSPACE_CONFIG` from the environment.
- **Timing covers the backward only:** the forward runs once. Each iteration then calls `torch.autograd.grad(out, inputs, grad_out, retain_graph=True)`. Using `autograd.grad` instead of `.backward()` avoids `.grad` accumulation kernels and keeps the timed region to the backward kernels alone.
- `torch.cuda.Event` timing: 10 warm-up iterations, then 50 timed iterations, with `synchronize()` before and after the timed loop. Before each timed iteration, a `torch.cuda._sleep` pad runs on the GPU. The pad lasts at least twice as long as the measured CPU launch time of one backward. The GPU stays busy while the CPU hands the backward to autograd's thread and launches it, so the events measure GPU kernel time and leave out launch latency. Without the pad, the smoke-sized matmul read 0.8 ms instead of 0.027 ms. The table shows the median; p90, min, max and the CPU launch time are in the JSON.
- **Bit identity:** 5 more backward runs on the same retained graph. Each run's gradients are hashed with SHA-256 over their raw bytes on the CPU. "yes (5/5)" means all 5 hashes match.
- Fixed seed (1337). Inputs are generated the same way in both modes. TF32 is off. bf16 reduced-precision reduction stays at the PyTorch default, which `env.json` records.
- An OOM is recorded as `OOM` and the run moves on. A crashed or timed-out worker is recorded as `ERR` for every shape it hadn't finished. Numbers are never filled in.
