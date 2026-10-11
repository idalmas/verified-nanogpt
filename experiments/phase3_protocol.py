"""
Phase 3: commit -> Fiat-Shamir -> sketches + openings, for a whole tiny-GPT step.

 1. honest runs: commitment size, proof size, prover overhead, verifier time, data seen
 2. attacks: post-commit tamper, choose-after-r with and without Fiat-Shamir, lying sketch
    vs row-opening rate, grinding cost
 3. exact F_p protocol for an int8 GEMM
Writes results/phase3_raw.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from freivalds.core import int_matmul_exact
from freivalds.protocol import (Transcript, challenges_for_gemm, prove, prove_int_gemm, verify,
                                verify_int_gemm)
from freivalds.tinygpt import (Config, Trace, get_batch, init_opt, init_params, load_shakespeare,
                               training_step)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 77
torch.manual_seed(SEED)
DATA = load_shakespeare()
RAW = {}


def make(cfg, B, T, tamper=None):
    P = init_params(cfg, seed=SEED)
    x, y = get_batch(DATA, B, T, torch.Generator().manual_seed(SEED))
    t = time.perf_counter()
    out = training_step(P, init_opt(P), x, y, cfg, tr=Trace(tamper=tamper or {}))
    return out[3], time.perf_counter() - t


def exp_honest():
    rows = []
    for d in [128, 256]:
        cfg = Config(d=d, n_head=4)
        tr, _ = make(cfg, 16, 64)
        P = init_params(cfg, seed=SEED)
        x, y = get_batch(DATA, 16, 64, torch.Generator().manual_seed(SEED))
        ts = []
        for _ in range(3):
            t = time.perf_counter(); training_step(P, init_opt(P), x, y, cfg, tr=Trace(record=False))
            ts.append(time.perf_counter() - t)
        t_step = float(np.median(ts))
        for rate in [0.01, 0.05, 0.2]:
            prove(tr, rate, rate)                      # warm layout-map caches
            S, troots, root, proof, tm = prove(tr, rate, rate)
            t = time.perf_counter()
            ok, fails, st = verify(S, troots, root, proof, rate, rate)
            t_ver = time.perf_counter() - t
            row = dict(d=d, rate=rate, ok=ok, t_step=t_step, t_commit=tm["t_commit"],
                       t_respond=tm["t_respond"], t_verify=t_ver, n_tensors=len(troots),
                       commitment_bytes=32, tensor_roots_bytes=36 * len(troots), **st)
            rows.append(row)
            print(f"[honest] d={d} rate={rate}: ok={ok} data={st['total_bytes']/1e6:.1f}MB "
                  f"proof={st['total']/1e6:.2f}MB (sketch {st['sketch_bytes']/1e6:.2f}, rows "
                  f"{st['opened_row_bytes']/1e6:.2f}, paths {st['path_bytes']/1e6:.2f}) "
                  f"seen={100*st['frac_bytes_seen']:.1f}% | step {t_step*1e3:.0f}ms commit "
                  f"{tm['t_commit']*1e3:.0f}ms respond {tm['t_respond']*1e3:.0f}ms verify {t_ver*1e3:.0f}ms")
    RAW["honest"] = rows


def exp_attacks():
    cfg = Config(d=128, n_head=4)
    rate = 0.05
    res = {}
    # (a) post-commit tamper: change one row of one big tensor after commitment
    tr, _ = make(cfg, 16, 64)
    caught = 0
    for s in range(5):
        def post(tensors, s=s):
            out = dict(tensors)
            big = [k for k, v in out.items() if v.dim() == 2 and v.shape[0] >= 1024]
            k = big[s % len(big)]
            v = out[k].clone(); v += 1e-3        # whole tensor shifted: every opened row differs
            out[k] = v
            return out
        S, tro, root, proof, _ = prove(tr, rate, rate, step=s, adversary={"post_commit": post})
        ok, fails, _ = verify(S, tro, root, proof, rate, rate, step=s)
        caught += (not ok) and any("Merkle" in f for f in fails)
    res["post_commit_caught_of_5"] = caught
    print(f"[attack] post-commit tamper caught {caught}/5")

    # (b) choose-after-r: perturbation orthogonal to the r of the honest commitment
    out = []
    for name in ["h0.fc1", "h1.qkv.dX", "h1.fc2.dW"]:
        for fs in [True, False]:
            tr0, _ = make(cfg, 16, 64)
            _, _, root0, _, _ = prove(tr0, rate, rate, fs=fs)
            g = next(x for x in tr0.gemms if x.name == name)
            r, *_ = challenges_for_gemm(Transcript(root0, 0, fs), name, tuple(g.A.shape), tuple(g.B.shape), rate)
            rb = torch.from_numpy(r[0, :, 0])
            u = torch.randn(rb.numel(), generator=torch.Generator().manual_seed(1), dtype=torch.float64)
            e = (u - (u @ rb) / (rb @ rb) * rb)
            scale = float(g.C.abs().max())          # perturbation as big as the largest entry
            e = e / e.abs().max() * scale

            def tam(C, e=e):
                C = C.clone(); C[7] += e.float(); return C
            tr1, _ = make(cfg, 16, 64, {name: tam})
            S1, tro1, root1, proof1, _ = prove(tr1, rate, rate, fs=fs)
            ok, fails, _ = verify(S1, tro1, root1, proof1, rate, rate, fs=fs)
            out.append(dict(gemm=name, fiat_shamir=fs, perturb_max=scale, gemm_check_failed=name in fails,
                            step_rejected=not ok, failing=fails))
            print(f"[attack] choose-after-r {name:10s} FS={fs}: perturbation max {scale:.3g} -> "
                  f"Freivalds check failed={name in fails}, step rejected={not ok} {fails}")
    res["choose_after_r"] = out

    # (c) lying sketch for one wrong row: detected iff that row of C is opened.
    #     Detection over 2000 transcripts, using the protocol's own challenge derivation,
    #     plus 3 full prove/verify runs per rate to confirm.
    tr, _ = make(cfg, 16, 64)
    g = next(x for x in tr.gemms if x.name == "h0.fc2")
    lie = []
    for rr in [0.01, 0.05, 0.2]:
        hits = 0
        for step in range(2000):
            T = Transcript(b"\x11" * 32, step)
            *_, rowsC = challenges_for_gemm(T, "h0.fc2", tuple(g.A.shape), tuple(g.B.shape), rr)
            hits += 5 in set(rowsC.tolist())
        full = 0
        for step in range(3):
            def tam(C):
                C = C.clone(); C[5] += 1.0; return C
            trt, _ = make(cfg, 16, 64, {"h0.fc2": tam})

            def liar(name, sk):
                if name == "h0.fc2":
                    sk = dict(sk); sk["w"] = sk["w"].copy(); sk["w"][0, 5] = sk["z"][0, 5]
                return sk
            S, tro, root, proof, _ = prove(trt, rr, rate, step=step, adversary={"sketch": liar})
            full += "h0.fc2" in verify(S, tro, root, proof, rr, rate, step=step)[1]
        lie.append(dict(row_rate=rr, detection=hits / 2000, transcripts=2000,
                        full_runs_detected=full, full_runs=3))
        print(f"[attack] lying sketch, one wrong row, row_rate={rr}: detection {hits/2000:.3f} "
              f"(full prove/verify detected {full}/3)")
    res["lying_sketch"] = lie

    # (d) grinding: re-commit (new nonce) until no lied row is opened
    rng = np.random.default_rng(SEED)
    grind = []
    m = 1024
    for rr in [0.05, 0.2]:
        for t_rows in [1, 10, 100]:
            attempts = []
            for run in range(30):
                lied = set(rng.choice(m, t_rows, replace=False).tolist())
                a = 0
                while a < 20000:
                    a += 1
                    T = Transcript(rng.bytes(32), 0)
                    *_, rowsC = challenges_for_gemm(T, "g", (m, 512), (512, 128), rr)
                    if not lied & set(rowsC.tolist()):
                        break
                attempts.append(a)
            grind.append(dict(row_rate=rr, lied_rows=t_rows, mean_attempts=float(np.mean(attempts)),
                              max_attempts=int(max(attempts)), capped=int(sum(a >= 20000 for a in attempts)),
                              theory=(1 - rr) ** (-t_rows)))
            print(f"[attack] grinding rate={rr} lied rows={t_rows}: mean attempts "
                  f"{np.mean(attempts):.1f} (theory ≈ {(1-rr)**(-t_rows):.1f}), capped {grind[-1]['capped']}/30")
    res["grinding"] = grind
    RAW["attacks"] = res


def exp_int():
    rng = np.random.default_rng(SEED)
    rows = []
    for n in [256, 1024]:
        A = rng.integers(-128, 128, (n, n)); B = rng.integers(-128, 128, (n, n))
        C = int_matmul_exact(A, B)
        for rr in [0.01, 0.1]:
            t = time.perf_counter(); pr = prove_int_gemm(A, B, C, row_rate=rr); tp = time.perf_counter() - t
            t = time.perf_counter(); ok = verify_int_gemm((A.shape, B.shape), *pr, row_rate=rr); tv = time.perf_counter() - t
            sk_bytes = sum(v.nbytes for v in pr[2].values())
            open_bytes = sum(len(b) + 32 * len(p) for b, p in pr[3].values())
            data_bytes = A.size * 4 + B.size * 4 + C.size * 8
            C2 = C.copy(); C2[3, 3] += 1
            caught_honest_sketch = not verify_int_gemm((A.shape, B.shape), *prove_int_gemm(A, B, C2, row_rate=rr), row_rate=rr)
            det = 0
            for s in range(40):
                # lie on z for the wrong row; caught iff row 3 of A is opened
                Cx = C.copy(); Cx[3, 3] += 1 + s
                det += not verify_int_gemm((A.shape, B.shape), *prove_int_gemm(A, B, Cx, row_rate=rr, lie_row=3), row_rate=rr)
            rows.append(dict(n=n, row_rate=rr, honest_ok=ok, t_prove=tp, t_verify=tv, sketch_bytes=sk_bytes,
                             opening_bytes=open_bytes, data_bytes=data_bytes,
                             wrong_C_honest_sketch_caught=caught_honest_sketch, lying_detected_of_40=det))
            print(f"[int] n={n} rate={rr}: ok={ok} prove {tp:.3f}s verify {tv:.3f}s proof "
                  f"{(sk_bytes+open_bytes)/1e6:.2f}MB of data {data_bytes/1e6:.1f}MB; wrong C w/ honest "
                  f"sketch caught={caught_honest_sketch}; lying caught {det}/40")
    RAW["int_protocol"] = rows


if __name__ == "__main__":
    t0 = time.time()
    exp_honest(); exp_attacks(); exp_int()
    RAW["wall_seconds"] = time.time() - t0
    json.dump(RAW, open(os.path.join(OUT, "phase3_raw.json"), "w"), indent=1, default=str)
    print(f"done in {time.time()-t0:.1f}s")
