"""
Shared helpers for the RQ semantic-ID diagnostic analysis (Yubao's Extension 2
task, see rq_code_analysis.md). All four analyses read precomputed RQ codes
already produced by Phase 1/2 eval jobs against the frozen official quantizer
(checkpoint/rq_clip_large.pth) -- no model loading needed here.

Code array layout: codes[:, 0] = modality token (0=image,1=text,2=image-text),
codes[:, 1:9] = the 8 semantic RQ levels (vocab 4096 each). All semantic-level
helpers below operate on codes[:, 1:] and are 1-indexed to match the
assignment's "RQ level 1..8" numbering.
"""
import os

import numpy as np

from data.preprocessing.utils import DATASET_CAN_NUM_UPPER_BOUND, DATASET_IDS, MBEIR_DATASET_TO_DOMAIN
from phase2_pool_utils import load_cand_pool

GEN_CODE_DIR_DEFAULT = "gen_code/GENIUS_t5small/Large/Instruct/InBatch"

# name -> whether the precomputed cand_pool mixes modalities and needs
# filtering to image-only candidates (same set phase2_prefix_overlap.py uses).
POOL_NAMES = {
    "mscoco_task0_test": False,
    "fashioniq_task7": False,
    "fashion200k_task0": True,
    "nights_task4": False,
    "cirr_task7": False,
    "visualnews_task0": True,
}

ID_TO_DATASET_NAME = {v: k for k, v in DATASET_IDS.items()}

# Maps our internal pool name -> canonical DATASET_IDS key, for labeling.
POOL_TO_DATASET_LABEL = {
    "mscoco_task0_test": "MSCOCO",
    "fashioniq_task7": "FashionIQ",
    "fashion200k_task0": "Fashion200K",
    "nights_task4": "NIGHTS",
    "cirr_task7": "CIRR",
    "visualnews_task0": "VisualNews",
}


def load_pool(gen_code_dir, name, image_only=None):
    """Load one precomputed candidate pool, optionally filtered to image-modality rows."""
    if image_only is None:
        image_only = POOL_NAMES.get(name, False)
    codes, ids = load_cand_pool(gen_code_dir, name)
    if image_only:
        mask = codes[:, 0] == 0
        codes, ids = codes[mask], ids[mask]
    return codes, ids


def load_all_pools(gen_code_dir):
    """Load every pool in POOL_NAMES, keyed by pool name -> (codes, ids)."""
    return {name: load_pool(gen_code_dir, name) for name in POOL_NAMES}


def dataset_id_from_ids(ids):
    return ids // DATASET_CAN_NUM_UPPER_BOUND


def split_by_modality(codes, ids):
    """Split a (codes, ids) pool by the modality token (codes[:, 0]).
    Returns {'image': (codes, ids), 'caption': (codes, ids)}, dropping any
    modality==2 (image-text pair) rows with a printed warning if present --
    used to isolate per-modality subsets of pools that mix modalities (e.g.
    MSCOCO's shared image+caption candidate pool)."""
    n_pair = int((codes[:, 0] == 2).sum())
    if n_pair:
        print(f"Warning: split_by_modality dropping {n_pair} modality==2 (image-text pair) rows.")
    image_mask = codes[:, 0] == 0
    caption_mask = codes[:, 0] == 1
    return {
        "image": (codes[image_mask], ids[image_mask]),
        "caption": (codes[caption_mask], ids[caption_mask]),
    }


def semantic_codes(codes):
    """Drop the modality token; returns the 8 semantic levels."""
    return codes[:, 1:]


def shared_prefix_length(a, b):
    """Vectorized longest-common-prefix length between two [N, L] semantic-code arrays."""
    mismatch = a != b
    has_mismatch = mismatch.any(axis=1)
    first_mismatch = np.argmax(mismatch, axis=1)
    return np.where(has_mismatch, first_mismatch, a.shape[1])


def hamming_distance(a, b):
    return (a != b).sum(axis=1)


def per_level_agreement(a, b):
    """Returns [L] array: fraction of pairs agreeing at each level."""
    return (a == b).mean(axis=0)


def output_dirs(root="/fnwi_fs/ivi/irlab/personal/tcetoje/rq_analysis"):
    dirs = {
        "root": root,
        "figures": os.path.join(root, "figures"),
        "tables": os.path.join(root, "tables"),
        "logs": os.path.join(root, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs
