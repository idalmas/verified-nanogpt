This is a project to demonstrate training verification of ML models, based on Karpathy NanoGPT. 

Unlike the original NanoGPT, training here is bit-deterministic. 

## Verifying a training step

With `--deterministic=True`, `train.py` saves a pair of snapshots every `iter_between_saves` iters: the state at iter t and at iter t+1, along with every micro batch iter t used. `verify.py` loads t, replays that one step with `step.training_step` (the same function `train.py` calls), and checks that the weights are bitwise equal to t+1.

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --deterministic=True --dropout=0.0 --max_iters=120 --iter_between_saves=100 --eval_interval=1000 --n_layer=2 --n_head=2 --n_embd=64 --block_size=64 --batch_size=8 --out_dir=out-det-a
python verify.py out-det-a/checkpoint:_100 out-det-a/checkpoint:_101
```

On CPU, add `--device=cpu --dtype=float32` to the training command. An optional third argument to `verify.py` replays on a different device, e.g. `cpu`. A bitwise match needs the same device the run trained on.

Deterministic mode forces one CPU thread, turns on `torch.use_deterministic_algorithms`, turns off TF32, and disables `torch.compile`. Training also stops any step that moves a weight by more than `max_weight_step_mult` × lr (default 10), which catches a model swapped in between steps.

### Results

Small model above (2 layers, 64-dim), replaying iter 100 → 101:

| Trained on | Replayed on | Grad accum | Result |
|---|---|---|---|
| CPU, float32 | CPU | 1 | `pair ok` |
| CPU, float32 | CPU | 4 | `pair ok` |
| H100, bf16 | H100 | 1 | `pair ok` |
| H100, bf16 | H100 | 4 | `pair ok` |
| H100, bf16 | CPU | 1 | `pair not ok` (expected: different kernels) |
| CPU and H100, with one weight in t+1 moved by 1 ulp | same device | 1 | `pair not ok` (expected) |

H100 runs used torch 2.8.0+cu128.

## What determinism costs: backward-kernel microbenchmark

[`benchmarks/kernel_bench.py`](benchmarks/kernel_bench.py) times the key backward-pass kernels at frontier-scale shapes in default and deterministic mode, and checks whether repeated backward passes are bitwise identical. Method: [`benchmarks/README.md`](benchmarks/README.md). Full table: [`benchmarks/results.md`](benchmarks/results.md). Raw timings: [`benchmarks/results.json`](benchmarks/results.json).

H100 PCIe, torch 2.8.0+cu128, cuDNN 9.10.2, bf16, median of 50 timed iters:

| Backward op | Deterministic overhead | Default bit-identical? |
|---|---|---|
| SDPA flash attention (what `model.py` uses) | 1.72–1.89× | no |
| SDPA memory-efficient attention | 1.45–1.55× | no, at 2 of 3 shapes |
| SDPA math attention | 1.00–1.02× (but 13–14× slower than flash) | yes |
| SDPA cuDNN attention | refuses deterministic mode (`No available kernel`) | no |
| `nn.Embedding` (Zipf token IDs) | 1.25–1.39× | yes |
| `nn.Linear` matmul (control) | 1.00–1.01× | yes |

Every kernel that ran in deterministic mode was bit-identical across 5 runs. Most of the cost is in attention backward. `flash_attn` was not installed, so it is untested.

## Acknowledgements

Built on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy and its contributors, which itself grew out of [minGPT](https://github.com/karpathy/minGPT). The model, data prep and training loop come from there, under the MIT license in [LICENSE](LICENSE).

## Freivalds-based step verification (experimental)

Cheaper-than-replay verification of a training step: Freivalds checks for every GEMM
(forward, input-gradient, weight-gradient), random coordinate sampling for non-linear ops, a
commit → Fiat–Shamir → sketch protocol, and an adversary that tries to implant a backdoor
while passing every check. CPU only, tiny models. Start with [`REPORT.md`](REPORT.md); per-phase
results are in [`results/`](results/), ideas in [`IDEAS.md`](IDEAS.md), the work journal in
[`LOG.md`](LOG.md).

```bash
pip install torch numpy matplotlib pytest
python data/shakespeare_char/prepare.py
pytest tests/                                   # 41 tests
python experiments/phase1_detection.py          # exact vs float Freivalds
python experiments/phase2_step.py               # sketch-verify a whole step
python experiments/phase3_protocol.py           # Merkle + Fiat–Shamir protocol
python experiments/phase4_adversary.py          # backdoor inside the tolerance (~1 h on 4 cores)
python experiments/phase5_hybrid.py             # Freivalds + bitwise row replay (~2 h+)
```

Headline: exact (integer / F_p) Freivalds rejected even a ±1 tamper on every step; float
Freivalds with the rigorous tolerance a bf16-output recipe needs let an adversary implant a
backdoor to 100% while all 1600 steps passed verification.
