#!/usr/bin/env python3
"""
Deterministic vs default backward-pass kernel benchmark.

Times the backward pass of attention, embedding and linear layers at
frontier-scale shapes, once with default (possibly nondeterministic) kernels and
once under torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG,
and checks whether repeated backward passes are bitwise identical.

Each (op group, mode) runs in its own subprocess, because the deterministic flag
and CUBLAS_WORKSPACE_CONFIG are process-global (the latter must be set before
CUDA initializes). Standalone: imports nothing from the rest of the repo.

    python benchmarks/kernel_bench.py           # full run (target: 1x H100)
    python benchmarks/kernel_bench.py --smoke   # tiny shapes, any CUDA GPU
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 1337
CUBLAS_DET = ":4096:8"
MODES = ("default", "deterministic")

# (batch, heads, seq, head_dim) / (vocab, dim, tokens) / (tokens, in, out)
FULL_SHAPES = {
    "attention": [(1, 64, 8192, 128), (4, 32, 4096, 128), (8, 16, 2048, 64)],
    "embedding": [(128256, 8192, 32768), (50304, 4096, 65536)],
    "matmul": [(32768, 8192, 8192), (32768, 8192, 28672), (8192, 4096, 4096)],
}
SMOKE_SHAPES = {
    "attention": [(2, 8, 1024, 64)],
    "embedding": [(8192, 256, 8192)],  # >3072 tokens so the same sort-based CUDA path as full shapes is used
    "matmul": [(2048, 1024, 1024)],
}
SHAPE_FIELDS = {
    "attention": ("B", "H", "S", "D"),
    "embedding": ("V", "dim", "T"),
    "matmul": ("T", "in", "out"),
}

# Worker groups: (op, group) -> impls benchmarked inside one subprocess.
GROUPS = {
    ("attention", "flash_attn"): ["flash_attn"],
    ("attention", "sdpa"): ["sdpa-auto", "sdpa-flash", "sdpa-efficient", "sdpa-cudnn", "sdpa-math"],
    ("embedding", "embedding"): ["nn.Embedding"],
    ("matmul", "linear"): ["nn.Linear"],
}

ZIPF_S = 1.0  # p(rank k) ∝ k^-s, roughly natural-language token frequencies


def shape_str(op, shape):
    return " ".join(f"{k}={v}" for k, v in zip(SHAPE_FIELDS[op], shape))


def percentile(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# ----------------------------------------------------------------------------
# Worker side (runs inside a subprocess with a single mode)

def setup_mode(torch, mode):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    if mode == "deterministic":
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_DET, "CUBLAS_WORKSPACE_CONFIG not set"
        torch.use_deterministic_algorithms(True)  # warn_only=False: nondeterministic ops raise
        torch.backends.cudnn.deterministic = True
    else:
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ, "CUBLAS_WORKSPACE_CONFIG leaked into default mode"
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


def hash_grads(torch, grads):
    h = hashlib.sha256()
    for g in grads:
        h.update(g.detach().contiguous().view(torch.uint8).cpu().numpy().data)
    return h.hexdigest()


def sleep_cycles_per_ms(torch):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(1_000_000)  # warm
    s.record()
    torch.cuda._sleep(10_000_000)
    e.record()
    torch.cuda.synchronize()
    return 10_000_000 / s.elapsed_time(e)


def time_backward(torch, bwd, warmup, iters):
    # Before each timed iteration a GPU sleep kernel is queued that outlasts the
    # CPU-side launch of the backward (autograd thread hand-off, Python), so the
    # GPU is never idle waiting on the CPU inside the timed region and the events
    # measure backward kernel time only.
    for _ in range(warmup):
        bwd()
    torch.cuda.synchronize()
    t = time.perf_counter()
    bwd()
    cpu_launch_ms = (time.perf_counter() - t) * 1e3
    torch.cuda.synchronize()
    pad_cycles = int(max(1.0, 2 * cpu_launch_ms) * sleep_cycles_per_ms(torch))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        torch.cuda._sleep(pad_cycles)
        starts[i].record()
        bwd()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)], cpu_launch_ms


def build_attention(torch, impl, shape, mode):
    import torch.nn.functional as F
    B, H, S, D = shape
    g = torch.Generator(device="cuda").manual_seed(SEED)
    kw = dict(device="cuda", dtype=torch.bfloat16, generator=g)
    # Generated in (B, H, S, D) for every impl so all impls see identical values.
    q, k, v, gout = (torch.randn(B, H, S, D, **kw) for _ in range(4))
    meta = {}
    if impl == "flash_attn":
        from flash_attn import flash_attn_func
        q, k, v, gout = (t.transpose(1, 2).contiguous() for t in (q, k, v, gout))  # (B, S, H, D)
        for t in (q, k, v):
            t.requires_grad_(True)
        out = flash_attn_func(q, k, v, causal=True, deterministic=(mode == "deterministic"))
        meta["flash_attn_deterministic_arg"] = mode == "deterministic"
    else:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        for t in (q, k, v):
            t.requires_grad_(True)
        name = impl.split("-", 1)[1]
        if name == "auto":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            enum = {"flash": "FLASH_ATTENTION", "efficient": "EFFICIENT_ATTENTION",
                    "cudnn": "CUDNN_ATTENTION", "math": "MATH"}[name]
            backend = getattr(SDPBackend, enum, None)
            if backend is None:
                raise NotImplementedError(f"SDPBackend.{enum} not in torch {torch.__version__}")
            with sdpa_kernel([backend]):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        meta["grad_fn"] = type(out.grad_fn).__name__
    return out, [q, k, v], gout, meta


def build_embedding(torch, impl, shape, mode):
    V, dim, T = shape
    gen = torch.Generator().manual_seed(SEED)
    probs = torch.arange(1, V + 1, dtype=torch.float64).pow(-ZIPF_S)
    ranks = torch.multinomial(probs, T, replacement=True, generator=gen)
    ids = torch.randperm(V, generator=gen)[ranks]  # frequent tokens scattered across the vocab
    counts = torch.bincount(ids, minlength=V)
    meta = {"zipf_s": ZIPF_S, "unique_tokens": int((counts > 0).sum()),
            "top_token_count": int(counts.max()), "top_token_frac": float(counts.max()) / T}
    g = torch.Generator(device="cuda").manual_seed(SEED)
    emb = torch.nn.Embedding(V, dim, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        emb.weight.copy_(torch.randn(V, dim, device="cuda", dtype=torch.bfloat16, generator=g))
    out = emb(ids.cuda())
    gout = torch.randn(T, dim, device="cuda", dtype=torch.bfloat16, generator=g)
    return out, [emb.weight], gout, meta


def build_linear(torch, impl, shape, mode):
    T, d_in, d_out = shape
    g = torch.Generator(device="cuda").manual_seed(SEED)
    kw = dict(device="cuda", dtype=torch.bfloat16, generator=g)
    lin = torch.nn.Linear(d_in, d_out, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(d_out, d_in, **kw) * d_in ** -0.5)
    x = torch.randn(T, d_in, **kw).requires_grad_(True)
    out = lin(x)
    gout = torch.randn(T, d_out, **kw)
    return out, [x, lin.weight], gout, {}


BUILDERS = {"attention": build_attention, "embedding": build_embedding, "matmul": build_linear}


def run_case(torch, op, impl, shape, mode, args):
    rec = {"op": op, "impl": impl, "shape": list(shape), "shape_str": shape_str(op, shape), "mode": mode}
    phase = "setup"
    caught = []
    try:
        if impl == "flash_attn":
            try:
                import flash_attn
                rec["flash_attn_version"] = flash_attn.__version__
            except ImportError as e:
                rec.update(status="unavailable", phase=phase, error=f"flash_attn not installed: {e}")
                return rec
        phase = "forward"
        # Capture warnings so e.g. "cuDNN SDPA is not deterministic" ends up next to the error it explains.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, inputs, gout, meta = BUILDERS[op](torch, impl, shape, mode)
        rec["meta"] = meta
        torch.cuda.synchronize()

        def bwd():
            return torch.autograd.grad(out, inputs, gout, retain_graph=True)

        phase = "backward-timing"
        times, cpu_launch_ms = time_backward(torch, bwd, args.warmup, args.iters)
        rec.update(median_ms=percentile(times, 0.5), p90_ms=percentile(times, 0.9),
                   min_ms=min(times), max_ms=max(times), times_ms=times, cpu_launch_ms=cpu_launch_ms,
                   warmup=args.warmup, iters=args.iters)

        phase = "bit-identity"
        hashes = []
        for _ in range(args.bit_runs):
            grads = bwd()
            hashes.append(hash_grads(torch, grads))
            del grads
        rec.update(hashes=hashes, distinct_hashes=len(set(hashes)), bit_identical=len(set(hashes)) == 1)
        rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rec["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="OOM", phase=phase, error=str(e)[:2000])
    except NotImplementedError as e:
        rec.update(status="unavailable", phase=phase, error=f"{type(e).__name__}: {e}"[:2000])
    except Exception as e:
        rec.update(status="error", phase=phase, error=f"{type(e).__name__}: {e}"[:2000])
    msgs = list(dict.fromkeys(str(w.message).strip() for w in caught))
    msgs.sort(key=lambda m: "determin" not in m.lower())  # the explanatory one first; stable otherwise
    if msgs:
        rec["warnings"] = msgs
        if rec.get("status") != "ok":
            rec["error"] += " | warnings: " + "; ".join(msgs)
    return rec


def run_worker(args):
    import torch
    impls = GROUPS[(args.op, args.group)]
    shapes = (SMOKE_SHAPES if args.smoke else FULL_SHAPES)[args.op]
    with open(args.out, "a") as f:
        def emit(rec):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            status = rec["status"]
            extra = f"median {rec['median_ms']:.3f} ms, bit-identical={rec['bit_identical']}" if status == "ok" \
                else rec.get("error", "").splitlines()[0][:160] if rec.get("error") else ""
            print(f"  [{args.mode:>13}] {args.op:<9} {rec['impl']:<15} {rec['shape_str']:<28} {status}: {extra}",
                  file=sys.stderr, flush=True)

        if not torch.cuda.is_available():
            for shape in shapes:
                for impl in impls:
                    emit({"op": args.op, "impl": impl, "shape": list(shape), "shape_str": shape_str(args.op, shape),
                          "mode": args.mode, "status": "error", "phase": "setup", "error": "CUDA not available"})
            return
        setup_mode(torch, args.mode)
        for shape in shapes:
            for impl in impls:
                torch.cuda.reset_peak_memory_stats()
                emit(run_case(torch, args.op, impl, shape, args.mode, args))
                gc.collect()
                torch.cuda.empty_cache()


def collect_env():
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        env.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                   cuda_available=torch.cuda.is_available(),
                   bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env.update(gpu_name=p.name, gpu_capability=f"{p.major}.{p.minor}",
                       gpu_memory_gb=round(p.total_memory / 1e9, 1), gpu_count=torch.cuda.device_count())
    except Exception as e:
        env["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        import flash_attn
        env["flash_attn"] = flash_attn.__version__
    except Exception as e:
        env["flash_attn"] = None
        env["flash_attn_error"] = f"{type(e).__name__}: {e}"
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        lines = r.stdout.strip().splitlines()
        env["nvidia_smi"] = lines or r.stderr.strip()
        if lines:
            env["driver_version"] = lines[0].split(",")[1].strip()
    except Exception as e:
        env["nvidia_smi"] = f"{type(e).__name__}: {e}"
    return env


# ----------------------------------------------------------------------------
# Orchestrator side (never touches CUDA itself)

def spawn(extra_args, mode=None, timeout=None):
    env = os.environ.copy()
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if mode == "deterministic":
        env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_DET
    cmd = [sys.executable, str(Path(__file__).resolve())] + extra_args
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=None if mode else subprocess.PIPE,
                          text=True, timeout=timeout)


def fmt_ms(r):
    if r is None:
        return "—"
    if r["status"] == "ok":
        return f"{r['median_ms']:.3f}"
    return {"OOM": "OOM", "unavailable": "N/A"}.get(r["status"], "ERR")


def fmt_bit(r):
    if r is None or r["status"] != "ok":
        return "—"
    n = len(r["hashes"])
    return f"yes ({n}/{n})" if r["bit_identical"] else f"**no** ({r['distinct_hashes']} distinct / {n})"


def build_markdown(records, env, cfg):
    by_key = {}
    order = []
    for r in records:
        key = (r["op"], r["impl"], r["shape_str"])
        if key not in by_key:
            by_key[key] = {}
            order.append(key)
        by_key[key][r["mode"]] = r

    lines = [
        "# Backward-pass kernel benchmark: default vs deterministic",
        "",
        f"GPU: {env.get('gpu_name', 'n/a')} · driver {env.get('driver_version', 'n/a')} · "
        f"torch {env.get('torch')} · CUDA {env.get('torch_cuda')} · cuDNN {env.get('cudnn')} · "
        f"flash_attn {env.get('flash_attn') or 'not installed'}",
        f"Run: {'smoke' if cfg['smoke'] else 'full'} · bf16 · TF32 off · {cfg['warmup']} warm-up + "
        f"{cfg['iters']} timed iters (median shown, p90 in results.json) · bit-identity over {cfg['bit_runs']} backward runs",
        "",
        "| Op | Impl | Shape | Default median ms | Det. median ms | Overhead (×) | Default bit-identical? | Det. bit-identical? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    notes = []
    for key in order:
        op, impl, shp = key
        d, t = by_key[key].get("default"), by_key[key].get("deterministic")
        overhead = f"{t['median_ms'] / d['median_ms']:.2f}" if d and t and d["status"] == t["status"] == "ok" else "—"
        label = impl
        if impl.startswith("sdpa"):
            def backend(fn):
                if not fn.startswith("ScaledDotProduct"):
                    return "math"  # math backend is composite; grad_fn is its last op
                return fn.replace("ScaledDotProduct", "").replace("Backward0", "")
            sel = [f"{m[:3]}→{backend(r['meta']['grad_fn'])}"
                   for m, r in (("default", d), ("det", t)) if r and r.get("meta", {}).get("grad_fn")]
            if sel:
                label += f" ({', '.join(sel)})"
        lines.append(f"| {op} | {label} | {shp} | {fmt_ms(d)} | {fmt_ms(t)} | {overhead} | {fmt_bit(d)} | {fmt_bit(t)} |")
        for m, r in (("default", d), ("deterministic", t)):
            if r and r["status"] != "ok":
                msg = (r.get("error") or "").strip().splitlines()[0][:300] if r.get("error") else ""
                notes.append(f"- {op} / {impl} / {shp} / {m}: **{r['status']}** ({r.get('phase', '?')}) — `{msg}`")
    if notes:
        lines += ["", "## Failures / unsupported", ""] + notes
    lines += ["", "SDPA labels show the backward actually recorded by autograd (grad_fn) in each mode. "
              "Timing = backward only, via torch.autograd.grad on a retained graph (forward runs once, outside the timed region)."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny shapes, finishes in about a minute")
    ap.add_argument("--ops", default="attention,embedding,matmul", help="comma-separated subset of ops to run")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--bit-runs", type=int, default=5)
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--timeout", type=int, default=None, help="per-subprocess timeout in seconds")
    # internal
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--collect-env", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--op", help=argparse.SUPPRESS)
    ap.add_argument("--group", help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.collect_env:
        print(json.dumps(collect_env()))
        return
    if args.worker:
        run_worker(args)
        return

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    timeout = args.timeout or (600 if args.smoke else 7200)
    cfg = {"smoke": args.smoke, "warmup": args.warmup, "iters": args.iters, "bit_runs": args.bit_runs,
           "seed": SEED, "zipf_s": ZIPF_S, "cublas_workspace_config_det": CUBLAS_DET}

    r = spawn(["--collect-env"], timeout=300)
    try:
        env = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        env = {"error": f"env collection failed (rc={r.returncode}): {(r.stderr or '')[-2000:]}"}
    (out_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    print(f"GPU: {env.get('gpu_name', 'n/a')}  torch {env.get('torch')}  flash_attn {env.get('flash_attn')}",
          file=sys.stderr)

    shapes_by_op = SMOKE_SHAPES if args.smoke else FULL_SHAPES
    records = []
    with tempfile.TemporaryDirectory() as td:
        for (op, group), impls in GROUPS.items():
            if op not in ops:
                continue
            for mode in MODES:
                out_file = Path(td) / f"{op}-{group}-{mode}.jsonl"
                wargs = ["--worker", "--op", op, "--group", group, "--mode", mode, "--out", str(out_file),
                         "--warmup", str(args.warmup), "--iters", str(args.iters), "--bit-runs", str(args.bit_runs)]
                if args.smoke:
                    wargs.append("--smoke")
                crash = None
                try:
                    p = spawn(wargs, mode=mode, timeout=timeout)
                    if p.returncode != 0:
                        crash = f"worker exited with code {p.returncode}"
                except subprocess.TimeoutExpired:
                    crash = f"worker timed out after {timeout}s"
                got = [json.loads(l) for l in out_file.read_text().splitlines()] if out_file.exists() else []
                done = {(g["impl"], g["shape_str"]) for g in got}
                for shape in shapes_by_op[op]:
                    for impl in impls:
                        if (impl, shape_str(op, shape)) not in done:
                            got.append({"op": op, "impl": impl, "shape": list(shape), "shape_str": shape_str(op, shape),
                                        "mode": mode, "status": "error", "phase": "worker",
                                        "error": crash or "no result recorded"})
                records += got

    (out_dir / "results.json").write_text(json.dumps({"env": env, "config": cfg, "records": records}, indent=2) + "\n")
    md = build_markdown(records, env, cfg)
    (out_dir / "results.md").write_text(md)
    print()
    print(md)
    print(f"Wrote {out_dir / 'results.md'}, results.json, env.json in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
