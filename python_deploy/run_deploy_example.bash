#!/usr/bin/env bash
set -euo pipefail

# Put your checkpoint at ./checkpoints/latest.ckpt first.
# Then update robot IP, camera serials, workspace bounds, and TCP settings in the config JSON.

python ./real_robot_ur12e_d405_speed_only/scripts/deploy_real_policy.py \
  --checkpoint ./checkpoints/latest.ckpt \
  --config ./real_robot_ur12e_d405_speed_only/configs/collect_pole_pickoff_laptop_remote.json \
  --num-trials 20 \
  --max-episode-s 180 \
  --control-hz 5 \
  --output-dir ./deploy_results/quganzi \
  --max-translation-per-step-m 0.015 \
  --max-rotation-per-step-rad 0.08
