"""
generate_pilot_configs.py

One-off script (not a training artifact): materializes the 6 concrete pilot
config files for the COCO balance-loss lambda/temperature calibration sweep,
each cloned from inbatch_coco_pilot.yaml with lambda/temperature/exp_name
patched in. Run once interactively before submitting any sbatch job, so every
file can be inspected before anything touches the cluster.

Usage:
    python generate_pilot_configs.py
"""

import os
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG_PATH = os.path.join(HERE, "inbatch_coco_pilot.yaml")
OUT_DIR = os.path.join(HERE, "_pilot_runs")

# (lambda, temperature) grid -- see plan doc for rationale behind each point
PILOT_RUNS = [
    (0.0, 100.0),
    (0.3, 100.0),
    (1.0, 100.0),
    (3.0, 100.0),
    (10.0, 100.0),
    (1.0, 10.0),  # saturation check: same lambda as above, 10x lower temperature
]


def fmt(x):
    # "0.3" not "0.30000001", "10" not "10.0" -- keeps filenames/exp_names tidy
    return f"{x:g}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = OmegaConf.load(BASE_CONFIG_PATH)

    for lam, temp in PILOT_RUNS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        # "lambda" is a Python reserved keyword -- cfg.balance_loss_config.lambda = lam
        # is a SyntaxError via attribute access, must use bracket/key-string access instead.
        cfg.balance_loss_config["lambda"] = lam
        cfg.balance_loss_config.temperature = temp
        cfg.experiment.exp_name = f"CocoPilotL{fmt(lam)}T{fmt(temp)}"

        out_name = f"pilot_l{fmt(lam)}_t{fmt(temp)}.yaml"
        out_path = os.path.join(OUT_DIR, out_name)
        OmegaConf.save(cfg, out_path)
        print(f"Wrote {out_path}  (lambda={lam}, temperature={temp}, exp_name={cfg.experiment.exp_name})")


if __name__ == "__main__":
    main()
