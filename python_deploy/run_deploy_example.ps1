$ErrorActionPreference = "Stop"

# 在新电脑上先把 checkpoint 放到 .\checkpoints\latest.ckpt，
# 再按实际相机 serial、机器人 IP、workspace 边界修改 config JSON。

python .\real_robot_ur12e_d405_speed_only\scripts\deploy_real_policy.py `
  --checkpoint .\checkpoints\latest.ckpt `
  --config .\real_robot_ur12e_d405_speed_only\configs\collect_pole_pickoff_laptop_remote.json `
  --num-trials 20 `
  --max-episode-s 180 `
  --control-hz 5 `
  --output-dir .\deploy_results\quganzi `
  --max-translation-per-step-m 0.015 `
  --max-rotation-per-step-rad 0.08 `
  --save-step-trace `
  --translation-ema-alpha 1.0 `
  --rotation-ema-alpha 1.0

# 如需测试轻度平滑，可改为例如：
#   --translation-ema-alpha 0.6 `
#   --rotation-ema-alpha 0.5 `
#   --translation-deadband-m 0.0015 `
#   --rotation-deadband-rad 0.01
