# MP1 真机部署包说明

这个文件夹是给新电脑部署测试用的最小代码包，不包含训练数据和 checkpoint。

## 目录结构

```text
部署/
  train_real.py
  MP1/
    setup.py
    mp1/
      config/
      policy/
      model/
      common/
      ...
  real_robot_ur12e_d405_speed_only/
    scripts/
      deploy_real_policy.py
      real_robot_utils.py
      collect_real_ur_speed_only.py
      ...
    configs/
      deploy_guagongzhuang_no_gripper_template.json
      collect_pole_pickoff_laptop_remote.json
      ...
  checkpoints/
    PUT_CKPT_HERE.txt
  deploy_results/
  requirements_deploy.txt
  check_deploy_env.py
  myenv_full_backup.yml
  run_deploy_example.ps1
  run_deploy_example.bash
```

## 必须额外准备

1. 训练好的 `.ckpt` 文件，例如 `latest.ckpt` 或 `epoch=0200.ckpt`。
2. 新电脑能连到 UR12e 控制器，并已安装 URCap/RTDE 相关运行环境。
3. 新电脑能识别两台 RealSense D405，相机 serial 要和 config JSON 一致。
4. Python 环境里安装 `requirements_deploy.txt` 中的包，以及匹配 CUDA 的 PyTorch。

## 推荐运行方式

把 checkpoint 放到：

```text
部署/checkpoints/latest.ckpt
```

进入部署目录：

```powershell
cd E:\.MP1\部署
```

先检查 Python 依赖：

```powershell
python .\check_deploy_env.py
```

先做 dry run 检查模型能否加载：

```powershell
python .\real_robot_ur12e_d405_speed_only\scripts\deploy_real_policy.py `
  --checkpoint .\checkpoints\latest.ckpt `
  --config .\real_robot_ur12e_d405_speed_only\configs\deploy_guagongzhuang_no_gripper_template.json `
  --dry-run `
  --num-trials 1
```

确认硬件连接和安全边界后再真机运行：

```powershell
.\run_deploy_example.ps1
```

## 需要重点修改的 config 项

```text
robot.ip
robot.tcp_offset
robot.payload_kg
robot.payload_cog
robot.home_joint_rad 或 robot.home_tcp_pose
robot.workspace_min
robot.workspace_max
cameras[*].serial
point_cloud.crop_min
point_cloud.crop_max
representation.action_mode
```

挂工装任务不涉及夹爪，应使用：

```json
"representation": {
  "obs_mode": "tcp_xyz_rot6d",
  "action_mode": "delta_tcp_pose"
}
```

如果 checkpoint 是旧的取杆/夹爪任务，通常是 7 维动作，应使用：

```json
"action_mode": "delta_tcp_pose_gripper"
```

部署脚本会检查 checkpoint 输出动作维度和 config 里的 `action_mode` 是否一致。不一致会直接报错，不会强行执行。

## 成功率记录方式

部署过程中按键：

```text
s      标记本次成功
f      标记本次失败
Space  中止本次，记为失败
q      退出测试
```

默认 `--max-episode-s 180`，也就是 3 分钟未完成自动算失败。每次 trial 的结果会写到 `deploy_results/`。

## 安全建议

1. 第一次只用 `--dry-run` 和 `--num-trials 1` 验证加载。
2. 第一次真机运行保留较小动作限制，例如 `--max-translation-per-step-m 0.015` 和 `--max-rotation-per-step-rad 0.08`。
3. 如果末端工具拆装后坐标方向变了，用 `--action-rotation-correction-rpy-deg R P Y` 做方向修正。
4. 机械臂碰撞、保护停、不可达、通信异常都会被脚本捕获并记为失败，同时尝试停止机器人。

## 这个包没有包含什么

```text
不包含训练 zarr 数据
不包含 data/outputs 训练输出
不包含 checkpoint，需要你手动放入 checkpoints/
不包含 RealSense/UR 官方驱动安装包
```
