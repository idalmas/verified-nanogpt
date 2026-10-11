"""
Binding the training data to the verified computation (closes Phase 9's batch-selection channel).

  1. Before training the lab commits to the dataset: a Merkle root over fixed-size token chunks
     (freivalds.protocol.MerkleTree).
  2. The batch for step t is *derived*, not chosen: start positions come from
     SHAKE-256(dataset_root ‖ t ‖ H(pre-step weights)). The pre-step weights are themselves the
     verified output of step t−1, so the lab cannot grind the order without changing verified
     state (and any change there is caught by the step verifier).
  3. The verifier recomputes the indices, opens the corresponding chunks against the dataset root
     (here: reads them from the committed array and checks their leaves), and compares with the
     batch the step used.
What it does not solve: poisoned content *inside* the committed dataset (a data-audit problem).
"""
import hashlib
import struct

import numpy as np
import torch

from freivalds.protocol import MerkleTree, verify_path

CHUNK = 4096


def commit_dataset(tokens):
    arr = np.asarray(tokens, dtype=np.int64)
    n = len(arr) // CHUNK
    rows = arr[: n * CHUNK].reshape(n, CHUNK)
    tree = MerkleTree(0xDA7A, rows)
    return tree, rows


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().contiguous().numpy().tobytes())
    return h.digest()


def derive_starts(root, step, shash, B, T, n_tokens):
    seed = hashlib.shake_256(b"batch-v1" + root + struct.pack("<Q", step) + shash).digest(8 * B)
    raw = np.frombuffer(seed, dtype=np.uint64)
    return (raw % np.uint64(n_tokens - T - 1)).astype(np.int64)


def batch_from_starts(tokens, starts, T):
    x = torch.stack([tokens[s:s + T] for s in starts])
    y = torch.stack([tokens[s + 1:s + 1 + T] for s in starts])
    return x, y


def check_batch(tree, rows, root, step, shash, batch, B, T):
    """Verifier side: recompute the prescribed batch, check its chunks against the root."""
    n_tokens = rows.size
    starts = derive_starts(root, step, shash, B, T, n_tokens)
    flat = torch.from_numpy(rows.reshape(-1))
    want = batch_from_starts(flat, starts, T)
    for s in starts:                       # Merkle openings of every chunk the batch touches
        for c in {int(s) // CHUNK, int(s + T) // CHUNK}:
            if not verify_path(0xDA7A, c, rows[c].tobytes(), tree.path(c), root):
                return False, "dataset opening failed"
    if not (torch.equal(want[0], batch[0]) and torch.equal(want[1], batch[1])):
        return False, "batch is not the one prescribed by the committed order"
    return True, "ok"
