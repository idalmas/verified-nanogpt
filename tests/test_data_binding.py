import torch

from freivalds.data_binding import (batch_from_starts, check_batch, commit_dataset, derive_starts,
                                    state_hash)
from freivalds.tinygpt import get_batch, load_shakespeare
from model import GPT, GPTConfig


def test_prescribed_batch_accepted_selected_batch_rejected():
    data = load_shakespeare()
    tree, rows = commit_dataset(data.numpy())
    root = tree.root
    torch.manual_seed(0)
    m = GPT(GPTConfig(block_size=32, vocab_size=65, n_layer=1, n_head=2, n_embd=32, dropout=0.0, bias=True))
    sh = state_hash(m)
    flat = torch.from_numpy(rows.reshape(-1))
    starts = derive_starts(root, 7, sh, 4, 32, rows.size)
    honest = batch_from_starts(flat, starts, 32)
    assert check_batch(tree, rows, root, 7, sh, honest, 4, 32)[0]
    # batch selection: the adversary uses any other batch (e.g. the best of K random ones)
    other = get_batch(data, 4, 32, torch.Generator().manual_seed(1))
    assert not check_batch(tree, rows, root, 7, sh, other, 4, 32)[0]
    # reordering: the same sequences in a different order
    swapped = (honest[0].flip(0), honest[1].flip(0))
    assert not check_batch(tree, rows, root, 7, sh, swapped, 4, 32)[0]
    # changed pre-step state → different prescribed batch
    with torch.no_grad():
        m.lm_head.weight.view(-1)[0] += 1e-3
    assert not check_batch(tree, rows, root, 7, state_hash(m), honest, 4, 32)[0]
