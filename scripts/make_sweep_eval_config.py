"""
Generic OmegaConf field-editor: load a base config, override a few fields, save to a new
path. Originally written for the COCO epoch sweep (swap in the checkpoint to evaluate,
optionally force candidate-pool code regeneration -- normally skipped, since the frozen RQ
quantizer produces the same candidate codes regardless of which T5 epoch is being
evaluated, and they're already cached on disk from a previous run).

Also reused for seed-variance config generation (RQ3 robustness addition): --seed and
--exp_name let this same script generate seed-variant train configs (pass --seed
--exp_name, omit --ckpt_name) and their matching eval configs (pass --exp_name
--ckpt_name, omit --seed -- the eval config's seed field is read but has no practical
effect since constrained_beam_search is fully deterministic, verified in
retriever.py:458-495). --ckpt_name is optional so train-config edits don't touch the
eval-only retrieval_config.cand_pools_config field, which doesn't exist in train configs.
"""
import argparse
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--out_config", required=True)
    parser.add_argument("--ckpt_name", default=None, help="Eval configs only")
    parser.add_argument("--gen_cand_codes", action="store_true", help="Eval configs only")
    parser.add_argument("--seed", type=int, default=None, help="Train configs only")
    parser.add_argument("--exp_name", default=None, help="Train or eval configs")
    args = parser.parse_args()

    config = OmegaConf.load(args.base_config)

    if args.ckpt_name is not None:
        config.model.ckpt_config.ckpt_name = args.ckpt_name
        config.retrieval_config.cand_pools_config.enable_gen_code = args.gen_cand_codes
    if args.seed is not None:
        config.seed = args.seed
    if args.exp_name is not None:
        config.experiment.exp_name = args.exp_name

    OmegaConf.save(config, args.out_config)
    print(f"Wrote {args.out_config} (ckpt_name={args.ckpt_name}, "
          f"gen_cand_codes={args.gen_cand_codes}, seed={args.seed}, exp_name={args.exp_name})")
