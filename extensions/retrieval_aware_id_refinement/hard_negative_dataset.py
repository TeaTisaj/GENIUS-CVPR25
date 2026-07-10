"""
Component 2 (retrieval-aware RQ refinement feasibility study) -- dataset wiring
for hard negatives.

Per the plan's corrected design (see the "Corrected design" paragraph, Component 2
of `.claude/plans/so-this-is-the-swirling-quiche.md`): `MBEIRDictInstructioneDataset`
(`data/mbeir_dataset.py` -- the SECOND class of that name; Python's later class
definition shadows the first one at that same name, and the second is the one
actually imported/used by `train.py`) is reused strictly *as-is*: no subclassing,
no new constructor args, no change to its `__getitem__` return shape. Appending
hard negatives into its existing `pool` dict was tried in an earlier draft of
this plan and found (on strict review) to break `mse_loss`/`cl_loss`/
`accuracy`/`accuracy_org`, all of which depend on strict 1:1 query:pool row
alignment (`residual_quantization.py:412-427` as of the plan's writing).

This wrapper instead returns the base dataset's untouched 4-tuple
`(query, pool, instruct, h_qids)` plus a 5th, separate key `hard_neg_pool`
(its own img_emb/txt_emb/img_mask/txt_mask dict), built via the same
`pool_dict`/`did_to_index` lookup the base class already performs -- read-only
reuse, the base class's internals are never mutated.
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class HardNegativeAugmentedDataset(Dataset):
    def __init__(self, base_dataset, hard_neg_dict, hard_neg_num):
        """
        Args:
            base_dataset: an already-constructed `MBEIRDictInstructioneDataset`
                instance, used exactly as it would be for ordinary (non-
                retrieval-aware) training -- this wrapper never modifies it.
            hard_neg_dict: {h_qid: [h_did, ...]} as produced by
                `mine_hard_negatives.py` (Component 1), for either the "hard"
                (frozen-teacher cosine-NN) or "random" (control) mining mode.
            hard_neg_num: fixed number of hard negatives per query (K) to draw
                from `hard_neg_dict[h_qid]` (must have >= hard_neg_num entries
                per queried h_qid; a query missing from the dict, or with too
                few mined negatives, raises loudly rather than silently
                truncating/padding, since that would indicate the mined-
                negatives file doesn't actually match this query split).
        """
        self.base_dataset = base_dataset
        self.hard_neg_dict = hard_neg_dict
        self.hard_neg_num = hard_neg_num
        # Read-only reuse of the base dataset's already-loaded pool embedding
        # dict + id-to-index map -- never mutated here.
        self.pool_dict = base_dataset.pool_dict
        self.did_to_index = base_dataset.did_to_index

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        query, pool, instruct, h_qids = self.base_dataset[index]
        # h_qids has shape [num_positives]. This wrapper's hard_neg_pool
        # construction below is only correct for num_positives == 1 (the
        # feasibility study's setting) -- assert rather than silently
        # mishandling a multi-positive batch this wrapper was never designed for.
        assert h_qids.numel() == 1, (
            f"HardNegativeAugmentedDataset assumes num_positives=1 (got {h_qids.numel()}); "
            "extend hard_neg_pool construction before using num_positives > 1."
        )
        h_qid = int(h_qids[0].item())

        neg_h_dids = self.hard_neg_dict.get(h_qid)
        assert neg_h_dids is not None, f"No mined hard negatives found for h_qid={h_qid}."
        assert len(neg_h_dids) >= self.hard_neg_num, (
            f"h_qid={h_qid} has only {len(neg_h_dids)} mined negatives, need {self.hard_neg_num}."
        )
        neg_h_dids = neg_h_dids[: self.hard_neg_num]

        neg_idxs = torch.tensor([self.did_to_index[h] for h in neg_h_dids], dtype=torch.long)
        hard_neg_pool = {
            "img_emb": self.pool_dict["img"][neg_idxs],
            "txt_emb": self.pool_dict["text"][neg_idxs],
            "img_mask": self.pool_dict["img_mask"][neg_idxs],
            "txt_mask": self.pool_dict["text_mask"][neg_idxs],
        }

        return query, pool, instruct, h_qids, hard_neg_pool


def hard_negative_collate_fn(batch):
    """Custom collate for HardNegativeAugmentedDataset's 5-tuple.

    `query`/`pool`/`instruct`/`h_qids` are collated via the default PyTorch
    collate (it already handles dicts-of-tensors and plain tensors correctly
    via recursive stacking) -- unchanged from what the base dataset would get
    without this wrapper.

    `hard_neg_pool` is handled manually: `default_collate` would stack the
    per-item `[hard_neg_num, dim]` (or `[hard_neg_num]` for masks) tensors
    into `[bs, hard_neg_num, dim]` (or `[bs, hard_neg_num]`); this collate
    additionally flattens that to `[bs * hard_neg_num, dim]` (or
    `[bs * hard_neg_num]`), matching the flattened-batch convention every other
    tensor in this pipeline already uses after `RQ.compute_single_batch`'s own
    `.view(-1, dim)` calls. `hard_neg_pool` is returned as a SEPARATE key, never
    merged into `pool` -- see Component 2's corrected design above.
    """
    queries = default_collate([item[0] for item in batch])
    pools = default_collate([item[1] for item in batch])
    instructs = default_collate([item[2] for item in batch])
    h_qids = default_collate([item[3] for item in batch])

    hard_neg_pools_stacked = default_collate([item[4] for item in batch])
    hard_neg_pool = {
        key: (value.reshape(-1, value.shape[-1]) if value.dim() == 3 else value.reshape(-1))
        for key, value in hard_neg_pools_stacked.items()
    }

    return queries, pools, instructs, h_qids, hard_neg_pool
