"""
Computes candidate_id -> full 8-level semantic code for the MSCOCO task0 candidate pool,
under the vanilla and strong Stage-1 RQ checkpoints. Standalone from
rq1_coco_variant_diagnostics.py because that script discards per-candidate IDs (only keeps
aggregate stats); this one needs the id->code mapping itself, to feed
tie_break_sensitivity.py's reconstruction of each query's top-predicted-code tie group from
the existing run files (see ECIR audit Round 5, item 7: does part of MSCOCO-strong's T->I
R@1 "gain" come from candidate-pool-order tie-breaking rather than genuine discrimination?).

Run with: PYTHONPATH=<repo>/src python compute_coco_candidate_codes.py --genir_dir <repo>
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.mbeir_dataset import MBEIRDictCandDataset
from models.residual_quantization.residual_quantization import RQ

VARIANTS = {
    "vanilla": "checkpoint/rq_clip_large/Large/Instruct/CocoVanilla/rq_clip_large_epoch_140.pth",
    "strong": "/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data/genius_checkpoints_stage1/rq_clip_large/Large/Instruct/CocoStrong/rq_clip_large_epoch_140.pth",
}
CAND_POOL_PATH = "cand_pool/local/mbeir_mscoco_task0_cand_pool.jsonl"
POOL_DICT_REL_PATH = "extracted_embed/CLIP_SF/cand/cand_pool_mscoco_task0_IT_dict.pt"


def resolve_checkpoint_path(genir_dir, ckpt_path):
    return ckpt_path if os.path.isabs(ckpt_path) else os.path.join(genir_dir, ckpt_path)


def load_quantizer(genir_dir, ckpt_path, device):
    ckpt = torch.load(resolve_checkpoint_path(genir_dir, ckpt_path), map_location="cpu", weights_only=False)
    model = RQ(config=ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def run_inference_with_ids(model, loader, device):
    all_codes, all_ids = [], []
    for pool, h_did in loader:
        img_mask = pool["img_mask"].view(-1).unsqueeze(-1).float().to(device, non_blocking=True)
        txt_mask = pool["txt_mask"].view(-1).unsqueeze(-1).float().to(device, non_blocking=True)
        img_emb = pool["img_emb"].view(-1, pool["img_emb"].size(-1)).float().to(device, non_blocking=True)
        txt_emb = pool["txt_emb"].view(-1, pool["txt_emb"].size(-1)).float().to(device, non_blocking=True)

        output = model.inference(img_emb, txt_emb, img_mask, txt_mask)
        code = output["code"]
        if code.ndim == 1:
            code = code.unsqueeze(0)

        all_codes.append(code[:, 1:].cpu().numpy())  # drop modality token, keep 8 semantic levels
        all_ids.append(h_did.numpy() if torch.is_tensor(h_did) else np.array(h_did))

    return np.concatenate(all_codes, axis=0), np.concatenate(all_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genir_dir", required=True)
    parser.add_argument("--mbeir_data_dir", default="/fnwi_fs/ivi/irlab/personal/tcetoje/mbeir_data")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        args.genir_dir, "extensions/rq2_balance_study/analysis/coco_candidate_codes"
    )
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pool_dict_dir = os.path.join(args.genir_dir, POOL_DICT_REL_PATH)
    assert os.path.exists(pool_dict_dir), f"Missing pool dict: {pool_dict_dir}"

    for variant, ckpt_path in VARIANTS.items():
        print(f"\n=== Variant: {variant} ===")
        dataset = MBEIRDictCandDataset(
            mbeir_data_dir=args.mbeir_data_dir,
            cand_pool_path=CAND_POOL_PATH,
            pool_dict_dir=pool_dict_dir,
            print_config=True,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        model = load_quantizer(args.genir_dir, ckpt_path, device)
        codes, ids = run_inference_with_ids(model, loader, device)
        print(f"  N candidates: {len(ids)}, codes shape: {codes.shape}")

        out_path = os.path.join(out_dir, f"{variant}_candidate_codes.npz")
        np.savez(out_path, ids=ids, codes=codes)
        print(f"  Saved: {out_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
