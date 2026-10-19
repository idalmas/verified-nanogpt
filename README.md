This is a project to demonstrate training verification of ML models, based on Karpathy NanoGPT. 

Unlike the original NanoGPT, training here is bit-deterministic. 

## Why this matters

Proposals for governing frontier AI often hinge on claims about training: how much compute a run used, what data it saw, and whether the released weights actually came out of that run. Today those claims are taken on trust. If a third party can cheaply re-execute pieces of a training run and check the results bit for bit, a developer's training claims become something an auditor or regulator can check instead of something they have to believe.

## Threat model

- **Prover:** the developer who trained the model. They publish checkpoints and claim "these weights came from running this code on this data."
- **Verifier:** an auditor, regulator or other lab that does not trust the prover but has the code, the checkpoints the prover hands over, and hardware of the same type.
- **Cheating we want to catch:** weights that did not come from the claimed training procedure. For example, a model trained on undisclosed data or with more compute than reported, then passed off as the output of a declared run; or a declared run with a different model swapped in partway through.

**What one replay proves.** `verify.py` shows that, given checkpoint t and the micro batches saved with it, one call to `training_step` produces exactly checkpoint t+1. That rules out tampering *within* that step: any change to the weights, optimizer state, batches or update rule between t and t+1 breaks the bitwise match.

**What it does not prove.** One step says nothing about the steps around it. A dishonest prover can train however they like, then fabricate a consistent pair by picking any weights as t, running one honest step to get t+1, and publishing both. Replay also does not check that the saved batches come from the claimed dataset, or that the batches are the ones an honest data loader would have drawn. Today the snapshots are saved at fixed, predictable iters (every `iter_between_saves`), so a prover knows in advance which steps will be looked at.

**How spot-checking raises the cost.** The protocol this code is a building block for:

1. The prover commits up front (e.g. with hashes) to a checkpoint at every step, or every k steps, plus the data order.
2. After training, the verifier picks step pairs at random and asks the prover to open them.
3. The verifier replays each opened segment and checks it matches the commitment.

If a fraction f of the steps were faked, the chance that n random spot checks all miss is about (1 − f)^n. Faking 1% of a run survives 300 checks only about 5% of the time. To get through, a cheater has to make nearly every committed step honest, which means actually running the claimed training. This repo implements step 3 for one step pair; the commitments and random challenges are not built yet.

## Related work

- **Proof-of-Learning** (Jia et al., [*Proof-of-Learning: Definitions and Practice*](https://arxiv.org/abs/2103.05633), IEEE S&P 2021) proposes that a trainer log intermediate weights and data, and a verifier re-executes randomly chosen segments to check them. That is the protocol sketched above. Follow-up work (e.g. Fang et al., [*Proof-of-Learning is Currently More Broken Than You Think*](https://arxiv.org/abs/2208.03567), 2023) shows that PoL's tolerance-based checks can be spoofed, because re-execution on GPUs is not exact and the verifier has to accept "close enough" weights. Bit-deterministic replay removes that tolerance: a match is exact or it fails.
- **Shavit, [*What does it take to catch a Chinchilla? Verifying Rules on Large-Scale Neural Network Training via Compute Monitoring*](https://arxiv.org/abs/2303.11341) (2023)** lays out a compute-governance framework in which chips log weight snapshots and a verifier replays sampled segments of training to confirm what was trained and on what data. This project is a small, concrete piece of that: it measures whether exact replay is achievable on real hardware and what determinism costs (see the benchmark below).

## Verifying a training step

## Verifying a training step

With `--deterministic=True`, `train.py` saves a pair of snapshots every `iter_between_saves` iters: the state at iter t and at iter t+1, along with every micro batch iter t used. `verify.py` loads t, replays that one step with `step.training_step` (the same function `train.py` calls), and checks that the weights are bitwise equal to t+1.

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --deterministic=True --dropout=0.0 --max_iters=120 --iter_between_saves=100 --eval_interval=1000 --n_layer=2 --n_head=2 --n_embd=64 --block_size=64 --batch_size=8 --out_dir=out-det-a
python verify.py out-det-a/checkpoint:_100 out-det-a/checkpoint:_101
```

On CPU, add `--device=cpu --dtype=float32` to the training command. An optional third argument to `verify.py` replays on a different device, e.g. `cpu`. A bitwise match needs the same device the run trained on.

Deterministic mode forces one CPU thread, turns on `torch.use_deterministic_algorithms`, turns off TF32, and disables `torch.compile`.

`train.py` also has a sanity check that stops any step that moves a weight by more than `max_weight_step_mult` × lr (default 10). It catches an accidental or crude swap of the model between steps, but it is **not a security check**: it runs on the prover's machine, so a dishonest prover can just turn it off, and even if it ran somewhere trusted, moving the weights toward a different model in many small steps stays under the bound. The bitwise replay is the check that matters.

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

## What's missing at scale

- **Multi-GPU and distributed training.** Verification has only been tested in a single process. `train.py` keeps nanoGPT's DDP code, but replaying a DDP step (whose saved batches are only rank 0's) is untested. Frontier runs use data, tensor and pipeline parallelism across thousands of GPUs, and a verifier would need to replay a step under the same sharding, or show that a different sharding gives the same bits.
- **Non-determinism from collective communication.** All-reduce in NCCL and similar libraries can sum gradients in an order that depends on topology, algorithm choice and timing. Floating-point addition is not associative, so the result can differ run to run. Bitwise replay needs fixed reduction orders, which costs throughput that is not measured here.
- **Cross-hardware verification.** Replay only matches on the device type that trained (the H100 → CPU row above fails as expected). So a verifier needs the same hardware, and probably the same driver, CUDA, cuDNN and PyTorch versions. That is a real cost and access barrier for auditors. Options include matching-hardware verification facilities, or tolerance-based checks across hardware, which bring back the spoofing problems that exact replay avoids.
- **Commitments and challenges.** As above: no hashing of checkpoints or data order, no random selection of which steps to open, and no check that saved batches come from the claimed dataset.
- **Scale of the test itself.** Results are on a 2-layer, 64-dim model. The kernel benchmark uses frontier-scale shapes, but no full large model has been replayed end to end.

## Acknowledgements

Built on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy and its contributors, which itself grew out of [minGPT](https://github.com/karpathy/minGPT). The model, data prep and training loop come from there, under the MIT license in [LICENSE](LICENSE).
