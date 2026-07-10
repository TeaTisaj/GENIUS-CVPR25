"""
Shared helpers for GENIUS Phase 2 (candidate-pool composition study).

Both phase2_pool_eval.py and phase2_prefix_overlap.py need to load and
concatenate precomputed candidate-pool RQ codes/ids without spinning up
torch.distributed or the full model — this module factors out that pattern,
which otherwise exists only inline inside
`generate_codes_for_config` (src/common/mbeir_generative_retriever.py).
"""
import os

import numpy as np


def load_cand_pool(gen_code_dir, name):
    """Load a precomputed candidate pool's RQ codes + ids.

    `name` matches the `mbeir_{name}_cand_pool_{codes,ids}.npy` naming
    convention already used under gen_code/.../cand_pool/.
    """
    codes_path = os.path.join(gen_code_dir, "cand_pool", f"mbeir_{name}_cand_pool_codes.npy")
    ids_path = os.path.join(gen_code_dir, "cand_pool", f"mbeir_{name}_cand_pool_ids.npy")
    codes = np.load(codes_path)
    ids = np.load(ids_path)
    return codes, ids


def assemble_pool(gen_code_dir, names):
    """Concatenate multiple precomputed candidate pools into one (codes, ids) pair."""
    codes_list = []
    ids_list = []
    for name in names:
        codes, ids = load_cand_pool(gen_code_dir, name)
        codes_list.append(codes)
        ids_list.append(ids)
    codes = np.concatenate(codes_list, axis=0)
    ids = np.concatenate(ids_list, axis=0)
    assert len(ids) == len(codes), "Mismatch between codes and ids length."
    assert len(set(ids.tolist())) == len(ids), "Duplicate ids across concatenated pools."
    return codes, ids
