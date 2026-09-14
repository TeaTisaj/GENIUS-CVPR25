#!/bin/bash
#SBATCH --job-name=dense_clipsf_raw
#SBATCH --partition=gpu
#SBATCH --nodelist=ilps-cn116
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/home/tcetoje/logs/dense_clip_sf_raw_baseline_%j.log
# Re-run of the raw CLIP-SF dense baseline (no RQ, no T5) so the paper's 55.5% T->I R@1
# reference has a persisted log (2026-09-14 senior-review fact-check). CPU scoring only.
source /home/tcetoje/miniconda3/etc/profile.d/conda.sh
conda activate genius2
cd /home/tcetoje/GENIUS-CVPR25/extensions/retrieval_aware_id_refinement
export PYTHONPATH=/home/tcetoje/GENIUS-CVPR25/src:/home/tcetoje/GENIUS-CVPR25/src/common
/home/tcetoje/miniconda3/envs/genius2/bin/python dense_clip_sf_raw_baseline.py --k_list 1 5 10
echo "EXIT CODE: $?"
