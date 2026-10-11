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
