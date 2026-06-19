# MP1 真实机器人全链路交接说明

本文档用于把当前 MP1 项目中和真实机器人实验相关的代码、脚本、论文路线、数据采集、数据清洗、训练、可视化、部署测试全部串起来。目标是：即使后续换一个大模型或换一个人接手，只要先读这份文档，就能知道相关文件在哪里、每个脚本做什么、当前有哪些数据、哪些配置不能混用。

最后更新时间：2026-04-22

工作区根目录：

```text
E:\.MP1
```

核心结论：

```text
旧 pole-pickoff / 取杆任务：含夹爪，state=[10]，action=[7]
新 guagongzhuang / 挂工装任务：不涉及夹爪，state=[9]，action=[6]
两类任务不能混用 task yaml、checkpoint、部署 action_mode。
```

## 1. 当前项目的主要目录

### 1.1 MP1 训练代码

```text
E:\.MP1\MP1
```

关键文件：

```text
E:\.MP1\train_real.py
E:\.MP1\MP1\train_real.py
E:\.MP1\MP1\mp1\config
E:\.MP1\MP1\mp1\config\task
E:\.MP1\MP1\mp1\dataset
E:\.MP1\MP1\mp1\policy
E:\.MP1\MP1\mp1\model\vision
```

说明：

```text
train_real.py 是真实 zarr 数据训练入口。
mp1/config/*.yaml 是方法级主配置。
mp1/config/task/*.yaml 是任务/数据集级配置。
mp1/dataset/real_multimodal_dataset.py 读取 global_image + wrist_image + point_cloud + state + action。
mp1/model/vision/real_multimodal_encoder.py 是真实多模态视觉编码器。
```

### 1.2 真实机器人采集、清洗、可视化、部署代码

```text
E:\.MP1\real_robot_ur12e_d405_speed_only
```

关键文件：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\collect_real_ur_speed_only.py
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\real_robot_utils.py
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\prepare_clean_real_dataset.py
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\prepare_multimodal_real_dataset.py
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\clean_all_real_datasets.py
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\clean_all_real_datasets_multimodal.py
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\render_real_episode_showcase.py
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\deploy_real_policy.py
```

说明：

```text
collect_real_ur_speed_only.py 负责真实 UR12e + D405 采集。
real_robot_utils.py 封装 UR RTDE、RealSense、夹爪、zarr/raw writer、状态/action 表示。
prepare_clean_real_dataset.py 清洗单个 zarr，支持 6 维无夹爪和 7 维含夹爪。
prepare_multimodal_real_dataset.py 清洗单个多模态 zarr，支持 global/wrist 图像 resize。
clean_all_real_datasets.py 批量清洗 data/zarr 或 data_all/zarr 下所有任务。
clean_all_real_datasets_multimodal.py 批量生成多模态 clean zarr。
render_real_episode_showcase.py 生成论文补充材料风格视频。
deploy_real_policy.py 加载 .ckpt 并在真机上部署测试成功率。
```

### 1.3 上传包

```text
E:\.MP1\UPLOAD_LATEST\MP1
```

用途：

```text
把需要上传服务器的 MP1 训练文件集中放在这里。
后续如果服务器没有最新 task yaml 或训练脚本，可以从这里按相对路径复制。
```

已有说明：

```text
E:\.MP1\UPLOAD_LATEST\MP1\REAL_ZARR_TRAINING_UPLOAD.md
E:\.MP1\UPLOAD_LATEST\MP1\REAL_DATA_PAPER_TUTORIAL.md
```

本文档也会复制到：

```text
E:\.MP1\UPLOAD_LATEST\MP1\MP1_REAL_ROBOT_FULL_CONTEXT_CN.md
```

### 1.4 论文目录

```text
C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略
```

关键文件：

```text
C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略\root.tex
C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略\sec\4_system_and_dataset.tex
C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略\sec\5_experiments.tex
C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略\multimedia
```

论文当前真实机器人路线：

```text
全局 RGB + 腕部 RGB + 腕部局部点云
也就是 asymmetric global/wrist perception。
```

论文里真实机器人主线不是 point-cloud only，也不是对称 96x96 双图像，而是非对称多模态。

## 2. 论文方法名和代码模式的对应关系

代码里训练脚本的 `MODE` 有三个：

```text
pisb
pa-pisb
pa-pisb-ib
```

大致对应关系：

```text
pisb       -> PISB baseline
pa-pisb    -> AP-PISB / axis-aware intermediate
pa-pisb-ib -> 当前代码中最接近论文完整方法的扰动增强版本
```

注意：

```text
pa-pisb-ib 更适合扰动真实机器人部署对比。
静态任务一般先跑 pa-pisb。
完整论文对比应在同一 observation interface 下跑 pisb / pa-pisb / pa-pisb-ib。
```

## 3. 三个真实训练脚本的区别

根目录下有三个真实数据训练快捷脚本：

```text
E:\.MP1\auto_run_real.bash
E:\.MP1\auto_run_real_multimodal.bash
E:\.MP1\auto_run_real_multimodal_asym.bash
```

### 3.1 `auto_run_real.bash`

输入：

```text
point_cloud + agent_pos
```

用途：

```text
点云-only 真实数据训练。
更接近公开仿真 benchmark 的老接口。
适合做点云基线，不是论文真实机器人主线。
```

典型命令：

```bash
bash ./auto_run_real.bash 3 pa-pisb real_some_task_clean_upload 0
```

### 3.2 `auto_run_real_multimodal.bash`

输入：

```text
global_image [3, 96, 96]
wrist_image  [3, 96, 96]
point_cloud  [512, 3]
agent_pos
```

用途：

```text
对称分辨率多模态。
适合作为节省显存或兼容旧实验的工程折中。
不是最贴论文的真实机器人主线。
```

典型命令：

```bash
bash ./auto_run_real_multimodal.bash 3 pa-pisb real_some_task_multimodal_clean_upload 0
```

### 3.3 `auto_run_real_multimodal_asym.bash`

输入：

```text
global_image [3, 128, 128]
wrist_image  [3, 96, 96]
point_cloud  [512, 3]
agent_pos
```

用途：

```text
论文主线。
全局相机保留更高分辨率，腕部相机保留局部视觉和局部点云。
```

典型命令：

```bash
bash ./auto_run_real_multimodal_asym.bash 3 pa-pisb real_some_task_multimodal_asym_clean_upload 0
```

### 3.4 脚本参数解释

以这个命令为例：

```bash
bash ./auto_run_real_multimodal_asym.bash 3 pa-pisb real_guagongzhuang_huangdong_multimodal_asym_clean_upload 0
```

参数含义：

```text
3                                                   -> 物理 GPU ID
pa-pisb                                             -> 方法模式
real_guagongzhuang_huangdong_multimodal_asym_clean_upload -> Hydra task config 名称
0                                                   -> seed
```

脚本内部最终执行的是：

```text
python train_real.py --config-name <方法主配置> task=<任务配置名> training.seed=<seed>
```

关键点：

```text
真正决定跑哪个任务的是 task=...
真正决定读哪个 zarr 的是 task yaml 里的 dataset.zarr_path
static / shake 只是旧 pole-pickoff 任务的快捷别名，多任务场景不要依赖它们。
```

## 4. 真实数据采集流程

### 4.1 采集脚本

采集入口：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\collect_real_ur_speed_only.py
```

采集配置示例：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\configs\collect_pole_pickoff_laptop_remote.json
E:\.MP1\real_robot_ur12e_d405_speed_only\configs\collect_vibration_grasp_laptop_remote.json
E:\.MP1\real_robot_ur12e_d405_speed_only\configs\speed_only_config_template.json
```

采集命令模板：

```bash
cd E:\.MP1\real_robot_ur12e_d405_speed_only
python scripts\collect_real_ur_speed_only.py --config configs\collect_pole_pickoff_laptop_remote.json --overwrite
```

如果只想演练流程，不连真机：

```bash
python scripts\collect_real_ur_speed_only.py --config configs\collect_pole_pickoff_laptop_remote.json --dry-run
```

### 4.2 采集时的控制逻辑

采集脚本使用 speed-only 键盘控制。

轨迹控制键：

```text
b     -> 自动回 home 后开始录制新轨迹
v     -> 保存当前轨迹为成功
n     -> 保存当前轨迹为失败
x     -> 丢弃当前轨迹
h     -> 未录制状态下手动回 home
Space -> 立即清零速度
q     -> 退出
```

TCP 速度控制键：

```text
w/s -> TCP X 方向速度
a/d -> TCP Y 方向速度
r/f -> TCP Z 方向速度
i/k -> TCP Rx 角速度
j/l -> TCP Ry 角速度
u/o -> TCP Rz 角速度
```

注意：

```text
键盘方向是采集时的人机控制方向，不等于训练数据最终一定要 7 维。
最终训练动作维度取决于清洗脚本输出的 action_mode。
```

### 4.3 原始数据结构

采集后会生成两类数据：

```text
raw episode 文件夹
zarr 数据集
```

典型目录：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data\raw\<task>\episode_0000
E:\.MP1\real_robot_ur12e_d405_speed_only\data\zarr\<task>\<task>.zarr
```

raw episode 内部结构：

```text
robot\tcp_pose.npy / tcp_pose.csv
robot\joint_positions.npy / joint_positions.csv
robot\action.npy / action.csv
robot\timestamp.npy / timestamp.csv
cameras\global_d405\rgb\*.png
cameras\global_d405\depth\*.npy
cameras\global_d405\point_cloud\*.ply
cameras\wrist_d405\rgb\*.png
cameras\wrist_d405\depth\*.npy
cameras\wrist_d405\point_cloud\*.ply
metadata.json
```

zarr 数据通常包含：

```text
data/action
data/state
data/point_cloud
data/tcp_pose
data/joint_positions
data/tcp_speed
data/gripper_fraction
data/gripper_target_fraction
data/camera_global_d405_img
data/camera_wrist_d405_img
data/camera_global_d405_point_cloud
data/camera_wrist_d405_point_cloud
meta/episode_ends
```

## 5. 已有真实数据集

### 5.1 旧取杆任务 `pole_pickoff`

位置：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data_all
```

静态原始数据：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data_all\zarr\pole_pickoff\pole_pickoff.zarr
```

扰动原始数据：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data_all\zarr\pole_pickoff_shake\pole_pickoff_shake.zarr
```

已清洗数据：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data_all\clean_zarr\pole_pickoff_clean.zarr
E:\.MP1\real_robot_ur12e_d405_speed_only\data_all\clean_zarr\pole_pickoff_multimodal_clean.zarr
E:\.MP1\real_robot_ur12e_d405_speed_only\data_all\clean_zarr\pole_pickoff_multimodal_asym_clean.zarr
E:\.MP1\real_robot_ur12e_d405_speed_only\data_all\clean_zarr\pole_pickoff_shake_clean.zarr
E:\.MP1\real_robot_ur12e_d405_speed_only\data_all\clean_zarr\pole_pickoff_shake_multimodal_clean.zarr
E:\.MP1\real_robot_ur12e_d405_speed_only\data_all\clean_zarr\pole_pickoff_shake_multimodal_asym_clean.zarr
```

格式：

```text
state=[10]
action=[7]
含 gripper 状态和 gripper target
适合 delta_tcp_pose_gripper
```

### 5.2 新挂工装任务 `guagongzhuang`

位置：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data
```

静态 raw：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data\raw\guagongzhuang
```

晃动 raw：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data\raw\guagongzhuang_huangdong
```

静态 zarr：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data\zarr\guagongzhuang\guagongzhuang.zarr
```

晃动 zarr：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data\zarr\guagongzhuang_huangdong\guagongzhuang_huangdong.zarr
```

采集统计：

```text
guagongzhuang:           32 条成功轨迹，原始 zarr 7879 步
guagongzhuang_huangdong: 30 条成功轨迹，原始 zarr 7199 步
```

注意：

```text
这个任务不涉及夹爪。
不要把它清洗成 state=[10] / action=[7]。
正确格式是 state=[9] / action=[6]。
```

当前已清洗输出：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\data\clean_zarr\guagongzhuang_clean.zarr
E:\.MP1\real_robot_ur12e_d405_speed_only\data\clean_zarr\guagongzhuang_multimodal_clean.zarr
E:\.MP1\real_robot_ur12e_d405_speed_only\data\clean_zarr\guagongzhuang_multimodal_asym_clean.zarr
E:\.MP1\real_robot_ur12e_d405_speed_only\data\clean_zarr\guagongzhuang_huangdong_clean.zarr
E:\.MP1\real_robot_ur12e_d405_speed_only\data\clean_zarr\guagongzhuang_huangdong_multimodal_clean.zarr
E:\.MP1\real_robot_ur12e_d405_speed_only\data\clean_zarr\guagongzhuang_huangdong_multimodal_asym_clean.zarr
```

当前 clean zarr 形状：

```text
guagongzhuang_multimodal_asym_clean.zarr
state        (7475, 9)
action       (7475, 6)
point_cloud  (7475, 512, 3)
global_image (7475, 3, 128, 128)
wrist_image  (7475, 3, 96, 96)

guagongzhuang_huangdong_multimodal_asym_clean.zarr
state        (6805, 9)
action       (6805, 6)
point_cloud  (6805, 512, 3)
global_image (6805, 3, 128, 128)
wrist_image  (6805, 3, 96, 96)
```

清洗保留比例：

```text
guagongzhuang:           7475 / 7847，keep_ratio 约 0.9526
guagongzhuang_huangdong: 6805 / 7169，keep_ratio 约 0.9492
```

## 6. 数据清洗逻辑

### 6.1 清洗目标

清洗做的事情：

```text
平滑 TCP 位姿序列
根据 TCP 平移、旋转和 gripper 变化判断 idle
裁掉轨迹首尾长时间停顿
默认 preserve 时间轴，不压缩中间动态节奏
根据相邻 TCP pose 重新计算训练 action
对极小平移和旋转增量做 deadband 置零
重建 state 为 tcp_xyz + rot6d，加不加 gripper 由参数决定
```

重要参数：

```text
translation_window=7
rotation_window=5
translation_idle_threshold_m=5e-4
rotation_idle_threshold_rad=5e-4
timeline_mode=preserve
trim_boundary_idle=True
translation_deadband_m=2e-4
rotation_deadband_rad=1e-3
rotation_delta_frame=base
```

### 6.2 含夹爪任务清洗命令

旧取杆任务使用 7 维动作：

```bash
cd E:\.MP1\real_robot_ur12e_d405_speed_only

python scripts\clean_all_real_datasets.py ^
  --root E:\.MP1\real_robot_ur12e_d405_speed_only\data_all ^
  --overwrite

python scripts\clean_all_real_datasets_multimodal.py ^
  --root E:\.MP1\real_robot_ur12e_d405_speed_only\data_all ^
  --overwrite
```

默认输出：

```text
state=[10]
action=[7]
```

### 6.3 无夹爪任务清洗命令

新挂工装任务必须使用 6 维动作：

```bash
cd E:\.MP1\real_robot_ur12e_d405_speed_only

python scripts\clean_all_real_datasets.py ^
  --root E:\.MP1\real_robot_ur12e_d405_speed_only\data ^
  --overwrite ^
  --output-action-mode delta_tcp_pose ^
  --no-include-gripper-state
```

生成对称多模态：

```bash
python scripts\clean_all_real_datasets_multimodal.py ^
  --root E:\.MP1\real_robot_ur12e_d405_speed_only\data ^
  --overwrite ^
  --output-action-mode delta_tcp_pose ^
  --no-include-gripper-state
```

生成论文主线非对称多模态：

```bash
python scripts\prepare_multimodal_real_dataset.py ^
  --input-zarr E:\.MP1\real_robot_ur12e_d405_speed_only\data\zarr\guagongzhuang\guagongzhuang.zarr ^
  --output-zarr E:\.MP1\real_robot_ur12e_d405_speed_only\data\clean_zarr\guagongzhuang_multimodal_asym_clean.zarr ^
  --overwrite ^
  --global-image-height 128 ^
  --global-image-width 128 ^
  --wrist-image-height 96 ^
  --wrist-image-width 96 ^
  --output-action-mode delta_tcp_pose ^
  --no-include-gripper-state
```

```bash
python scripts\prepare_multimodal_real_dataset.py ^
  --input-zarr E:\.MP1\real_robot_ur12e_d405_speed_only\data\zarr\guagongzhuang_huangdong\guagongzhuang_huangdong.zarr ^
  --output-zarr E:\.MP1\real_robot_ur12e_d405_speed_only\data\clean_zarr\guagongzhuang_huangdong_multimodal_asym_clean.zarr ^
  --overwrite ^
  --global-image-height 128 ^
  --global-image-width 128 ^
  --wrist-image-height 96 ^
  --wrist-image-width 96 ^
  --output-action-mode delta_tcp_pose ^
  --no-include-gripper-state
```

### 6.4 清洗脚本最近做过的修改

修改文件：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\prepare_clean_real_dataset.py
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\clean_all_real_datasets.py
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\prepare_multimodal_real_dataset.py
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\clean_all_real_datasets_multimodal.py
```

新增能力：

```text
--output-action-mode delta_tcp_pose
--output-action-mode delta_tcp_pose_gripper
--no-include-gripper-state
```

原因：

```text
旧 pole-pickoff 是抓取/夹爪任务，需要 7 维动作。
新 guagongzhuang 是挂工装任务，不涉及夹爪，必须保持 6 维动作。
```

## 7. 训练配置

### 7.1 方法主配置

真实训练方法主配置位于：

```text
E:\.MP1\MP1\mp1\config
```

常用主配置：

```text
pisb-real-zarr-upload.yaml
pa-pisb-real-zarr-upload.yaml
pa-pisb-inertial-bridge-real-zarr-upload.yaml
pisb-real-multimodal-zarr-upload.yaml
pa-pisb-real-multimodal-zarr-upload.yaml
pa-pisb-inertial-bridge-real-multimodal-zarr-upload.yaml
pisb-real-multimodal-asym-zarr-upload.yaml
pa-pisb-real-multimodal-asym-zarr-upload.yaml
pa-pisb-inertial-bridge-real-multimodal-asym-zarr-upload.yaml
```

说明：

```text
方法主配置决定 policy 类、horizon、batch size、optimizer、checkpoint 规则。
任务配置决定 obs/action shape 和 zarr_path。
```

### 7.2 旧取杆任务 task yaml

位置：

```text
E:\.MP1\MP1\mp1\config\task
```

静态：

```text
real_pole_pickoff_clean_upload.yaml
real_pole_pickoff_multimodal_clean_upload.yaml
real_pole_pickoff_multimodal_asym_clean_upload.yaml
```

扰动：

```text
real_pole_pickoff_shake_clean_upload.yaml
real_pole_pickoff_shake_multimodal_clean_upload.yaml
real_pole_pickoff_shake_multimodal_asym_clean_upload.yaml
```

这些配置应保持：

```text
agent_pos shape: [10]
action shape: [7]
```

### 7.3 新挂工装任务 task yaml

新增配置：

```text
E:\.MP1\MP1\mp1\config\task\real_guagongzhuang_multimodal_asym_clean_upload.yaml
E:\.MP1\MP1\mp1\config\task\real_guagongzhuang_huangdong_multimodal_asym_clean_upload.yaml
```

这些配置必须保持：

```text
agent_pos shape: [9]
action shape: [6]
```

配置内容要点：

```yaml
global_image:
  shape: [3, 128, 128]
wrist_image:
  shape: [3, 96, 96]
point_cloud:
  shape: [512, 3]
agent_pos:
  shape: [9]
action:
  shape: [6]
```

服务器上需要改的字段：

```text
dataset.zarr_path
```

本地写的是服务器预期路径示例：

```text
/home/amax/cqy/MP1/MP1/data/clean_zarr/guagongzhuang_multimodal_asym_clean.zarr
/home/amax/cqy/MP1/MP1/data/clean_zarr/guagongzhuang_huangdong_multimodal_asym_clean.zarr
```

如果服务器实际数据目录不同，只改 `zarr_path`，不要改 shape。

### 7.4 新挂工装训练命令

静态挂工装：

```bash
bash ./auto_run_real_multimodal_asym.bash 3 pa-pisb real_guagongzhuang_multimodal_asym_clean_upload 0
```

晃动挂工装：

```bash
bash ./auto_run_real_multimodal_asym.bash 3 pa-pisb real_guagongzhuang_huangdong_multimodal_asym_clean_upload 0
```

晃动任务如果要跑扰动增强方法：

```bash
bash ./auto_run_real_multimodal_asym.bash 3 pa-pisb-ib real_guagongzhuang_huangdong_multimodal_asym_clean_upload 0
```

显式 Hydra 形式：

```bash
CUDA_VISIBLE_DEVICES=3 python train_real.py \
  --config-name pa-pisb-real-multimodal-asym-zarr-upload \
  task=real_guagongzhuang_multimodal_asym_clean_upload \
  training.device=cuda:0 \
  training.seed=0 \
  exp_name=pa_pisb_guagongzhuang_asym_seed0
```

```bash
CUDA_VISIBLE_DEVICES=3 python train_real.py \
  --config-name pa-pisb-inertial-bridge-real-multimodal-asym-zarr-upload \
  task=real_guagongzhuang_huangdong_multimodal_asym_clean_upload \
  training.device=cuda:0 \
  training.seed=0 \
  exp_name=pa_pisb_ib_guagongzhuang_huangdong_asym_seed0
```

### 7.5 训练输出

训练会输出 `.ckpt`：

```text
outputs/<date>/<time>/checkpoints/latest.ckpt
outputs/<date>/<time>/checkpoints/epoch=...-val_loss=....ckpt
outputs/<date>/<time>/checkpoints/epoch=0200.ckpt
outputs/<date>/<time>/checkpoints/epoch=0400.ckpt
```

checkpoint 规则：

```text
checkpoint.save_ckpt: True       -> 必须打开，否则不会保存 ckpt
training.checkpoint_every: 50    -> 每 50 个 epoch 更新 latest.ckpt 和 top-k val_loss ckpt
checkpoint.save_periodic_every: 200 -> 每完成 200 个 epoch 额外保留一个命名 ckpt，不覆盖
```

如果想每 300 个 epoch 保留一个命名 ckpt，可以在命令行加：

```bash
checkpoint.save_periodic_every=300
```

注意：`latest.ckpt` 会被覆盖；`epoch=0200.ckpt`、`epoch=0400.ckpt` 这种周期 ckpt 不会被覆盖。

路径注意：

```text
如果从外层根目录的 train_real.py 启动，工作目录应为项目外层根目录，例如 E:\.MP1。
如果从内层 MP1\train_real.py 启动，工作目录也应为项目外层根目录。
因此 ckpt 应该落在 E:\.MP1\data\outputs\...，而不是 E:\data\outputs\...。
入口脚本已经做了兼容修正：外层 train_real.py 会把 import 路径指到 MP1\mp1，同时把工作目录固定为外层项目根目录。
```

真实任务的 `env_runner` 是 `null`，因此真实成功率不会在训练阶段自动得到。

真实训练阶段主要看：

```text
train_loss
val_loss
sample_loss
checkpoint top-k by val_loss
```

要得到真实成功率，必须部署到真机跑 trial。

## 8. 可视化视频

### 8.1 可视化脚本

入口：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\render_real_episode_showcase.py
```

作用：

```text
读取 raw episode 文件夹。
显示 global RGB、wrist RGB、wrist-local point cloud、3D robot workspace 示意。
输出 MP4，风格类似论文补充材料。
```

### 8.2 挂工装任务可视化语义

刚刚修正过的重要语义：

```text
挂工装任务中，只有机械臂相关的东西会晃。
目标物是固定的白色母线/线缆，不随晃动平台移动。
母线方向平行于机械臂 Y 轴。
视频里的目标不能画成绿色杆，也不能方向错误。
工装需要画出来，用一个简化“方框主体 + C 字形卡口”模型表示，并随 TCP/机械臂侧运动。
工装在 3D 示意图里应保持世界 Z 方向竖直，C 字形卡口平面与沿 Y 方向的母线垂直。
C 口不能画反，母线应从开口侧进入卡口，不能视觉上穿过 C 口的封闭墙壁。
```

修改文件：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\render_real_episode_showcase.py
```

新增/修改点：

```text
Fixture hanging 任务检测
固定白色 bus cable 模型
随 TCP 平移、保持竖直的 fixture/hanger 简化模型
方框主体 + C 字形卡口；卡口开口方向已校正，比例相对机械臂缩小
disturbed 语义改成 robot-side sway only, target cable fixed
guagongzhuang_huangdong 识别为 disturbed
```

特别注意：

```text
guagongzhuang 静态集的 metadata.task_id 已批量修正为“挂工装”。
guagongzhuang_huangdong 的 metadata.task_id 保持为“挂工装_晃动”。
可视化脚本仍保留 episode 父目录 guagongzhuang / 中文 task_name 兜底识别，避免旧 metadata 再导致误走取杆分支。
```

### 8.3 当前已生成视频

输出目录：

```text
C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略\multimedia
```

已生成：

```text
guagongzhuang_episode_0031_showcase.mp4
guagongzhuang_huangdong_episode_0029_showcase.mp4
ral_showcase_guagongzhuang.mp4
```

原论文旧视频：

```text
ral_showcase.mp4
pole_pickoff_shake_episode_0029_showcase.mp4
```

2026-04-22 新拍摄的晃动平台参数和真实图像也在：

```text
C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略\multimedia
```

文件包括：

```text
080c765ae8d910b1cfebc5b89629e73a.jpg
c24e7af4362cd33310c56286af3bc1c9.jpg
d2b47a802acb56aae235eda84c22b330.jpg
42dee84d450f78694b166fa04a26e4b9.mp4
6da25db7e5bed9f92b5c1863f8bce231.mp4
```

### 8.4 重新生成视频命令

静态挂工装：

```bash
cd E:\.MP1\real_robot_ur12e_d405_speed_only

python scripts\render_real_episode_showcase.py ^
  --episode-dir E:\.MP1\real_robot_ur12e_d405_speed_only\data\raw\guagongzhuang\episode_0031 ^
  --output-mp4 "C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略\multimedia\guagongzhuang_episode_0031_showcase.mp4" ^
  --mode preserve ^
  --layout supp_clean
```

晃动挂工装：

```bash
python scripts\render_real_episode_showcase.py ^
  --episode-dir E:\.MP1\real_robot_ur12e_d405_speed_only\data\raw\guagongzhuang_huangdong\episode_0029 ^
  --output-mp4 "C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略\multimedia\guagongzhuang_huangdong_episode_0029_showcase.mp4" ^
  --mode preserve ^
  --layout supp_clean
```

当前机器没有 `ffmpeg`，所以合并视频用 OpenCV 脚本做。

## 9. 部署测试成功率

### 9.1 部署脚本

入口：

```text
E:\.MP1\real_robot_ur12e_d405_speed_only\scripts\deploy_real_policy.py
```

作用：

```text
加载 train_real.py 产生的 .ckpt。
恢复 cfg、policy 权重和 normalizer。
连接 UR RTDE 和 RealSense。
实时构造 observation buffer。
调用 policy.predict_action。
把预测 delta action 发送给真实机器人。
逐 trial 记录人工成功/失败、超时、异常。
输出 JSONL 和 summary JSON。
```

### 9.2 为什么需要 `.ckpt`，不推荐普通 `.pth`

部署需要：

```text
policy 权重
训练 cfg
normalizer
shape_meta
n_obs_steps
action_dim
```

`train_real.py` 的 `.ckpt` 包含这些信息。

普通 `.pth` 如果只是 `state_dict`，通常缺少 cfg 和 normalizer，脚本会拒绝部署。

正确 checkpoint：

```text
latest.ckpt
top-k checkpoint .ckpt
```

### 9.3 部署命令模板

```bash
cd E:\.MP1\real_robot_ur12e_d405_speed_only

python scripts\deploy_real_policy.py ^
  --checkpoint E:\path\to\latest.ckpt ^
  --config configs\collect_pole_pickoff_laptop_remote.json ^
  --num-trials 20 ^
  --max-episode-s 180 ^
  --control-hz 5 ^
  --output-dir deploy_results\guagongzhuang
```

如果先 dry-run：

```bash
python scripts\deploy_real_policy.py ^
  --checkpoint E:\path\to\latest.ckpt ^
  --config configs\collect_pole_pickoff_laptop_remote.json ^
  --num-trials 2 ^
  --dry-run ^
  --output-dir deploy_results\dry_run
```

### 9.4 成功率如何判断

当前最可靠方式：

```text
人工判断成功/失败。
```

运行中按键：

```text
s     -> 当前 trial 成功
f     -> 当前 trial 失败
Space -> 中止，记失败
q     -> 退出评测
```

原因：

```text
真实任务成功往往涉及“是否挂上”“是否掉落”“是否明显碰撞”“是否完成目标状态”。
如果没有额外视觉检测、力传感器、接触传感器或工装状态传感器，程序不应该自动猜成功。
```

默认超时：

```text
180 秒
```

如果 3 分钟没完成：

```text
自动停止并记 failure，reason=timeout。
```

### 9.5 碰撞、不可达、卡死怎么办

部署脚本处理策略：

```text
UR/RTDE 抛异常 -> safe_stop_robot -> 当前 trial failure
不可达 / moveL 报错 -> safe_stop_robot -> 当前 trial failure
人工发现危险 -> 按 Space -> 当前 trial failure
进入 protective stop -> 人工在 UR 示教器/控制端恢复，脚本不应自动解锁
```

原因：

```text
真机安全优先级高于连续评测。
保护停机后自动恢复继续跑风险很高，不建议写进脚本。
```

### 9.6 末端工具拆装导致 X+ 方向不一致怎么办

首选方案：

```text
重新标定 TCP 和工具坐标。
更新部署 JSON 中 robot.tcp_offset。
保证采集、训练、部署时的 TCP 坐标定义一致。
```

临时方案：

```bash
--action-rotation-correction-rpy-deg 0 0 180
```

这个参数会对 policy 输出的平移和旋转增量做一个固定旋转修正。

注意：

```text
这是临时补偿，不是正式标定。
如果 TCP offset、相机外参、工具安装方向都变了，只修 action 方向可能仍然不够。
正式实验必须保证坐标系一致。
```

### 9.7 无夹爪部署配置

新挂工装任务部署时，JSON 里必须是：

```json
"representation": {
  "obs_mode": "tcp_xyz_rot6d",
  "action_mode": "delta_tcp_pose"
}
```

不能用：

```json
"action_mode": "delta_tcp_pose_gripper"
```

如果错用，部署脚本会检查出：

```text
policy action_dim=6
robot action_mode expects 7
```

然后直接报错，不会执行机器人动作。

## 10. 真实机器人配置 JSON

配置文件控制：

```text
robot IP
TCP offset
payload
home_joint_rad
home_tcp_pose
workspace_min/max
rotation_delta_frame
gripper
camera serial
point cloud crop
collection sample_hz
speed_control control_hz
```

典型片段：

```json
"representation": {
  "obs_mode": "tcp_xyz_rot6d",
  "action_mode": "delta_tcp_pose"
}
```

对有夹爪任务：

```json
"action_mode": "delta_tcp_pose_gripper"
```

对无夹爪任务：

```json
"action_mode": "delta_tcp_pose"
```

工作空间边界：

```json
"workspace_min": [0.3, -0.18, 0.03],
"workspace_max": [0.6, 0.18, 0.32]
```

注意：

```text
部署前必须确认 workspace_min/max 覆盖任务区域但不允许撞桌/撞线缆/撞工装。
```

## 11. 数据维度和配置一致性检查

### 11.1 旧取杆任务

clean zarr：

```text
state        (N, 10)
action       (N, 7)
point_cloud  (N, 512, 3)
global_image (N, 3, 128, 128) for asym
wrist_image  (N, 3, 96, 96)
```

task yaml：

```yaml
agent_pos:
  shape: [10]
action:
  shape: [7]
```

部署 JSON：

```json
"action_mode": "delta_tcp_pose_gripper"
```

### 11.2 新挂工装任务

clean zarr：

```text
state        (N, 9)
action       (N, 6)
point_cloud  (N, 512, 3)
global_image (N, 3, 128, 128) for asym
wrist_image  (N, 3, 96, 96)
```

task yaml：

```yaml
agent_pos:
  shape: [9]
action:
  shape: [6]
```

部署 JSON：

```json
"action_mode": "delta_tcp_pose"
```

### 11.3 最常见错误

错误 1：

```text
用 pole_pickoff task yaml 训练 guagongzhuang zarr。
```

结果：

```text
shape 不匹配，或训练出错误 action_dim。
```

错误 2：

```text
无夹爪任务清洗时忘记 --output-action-mode delta_tcp_pose。
```

结果：

```text
action 被变成 7 维，多出无意义 gripper target。
```

错误 3：

```text
部署 6 维 policy 时 JSON 仍写 delta_tcp_pose_gripper。
```

结果：

```text
部署脚本拒绝执行，或如果没有检查会导致动作解释错误。
```

错误 4：

```text
服务器 zarr_path 没改。
```

结果：

```text
训练找不到 zarr。
```

错误 5：

```text
末端工具拆装后没有重新标定 TCP。
```

结果：

```text
policy 输出方向和真实动作方向不一致，成功率会大幅下降，并可能有碰撞风险。
```

## 12. 需要上传服务器的内容

训练相关：

```text
auto_run_real.bash
auto_run_real_multimodal.bash
auto_run_real_multimodal_asym.bash
train_real.py
mp1/dataset/real_multimodal_dataset.py
mp1/model/vision/real_multimodal_encoder.py
mp1/model/vision/obs_encoder_factory.py
mp1/policy/*.py 中相关 PISB / PA-PISB / IB 文件
mp1/config/*real*.yaml
mp1/config/task/real_*.yaml
```

新增挂工装 task yaml：

```text
mp1/config/task/real_guagongzhuang_multimodal_asym_clean_upload.yaml
mp1/config/task/real_guagongzhuang_huangdong_multimodal_asym_clean_upload.yaml
```

真实部署相关：

```text
real_robot_ur12e_d405_speed_only/scripts/deploy_real_policy.py
real_robot_ur12e_d405_speed_only/scripts/real_robot_utils.py
real_robot_ur12e_d405_speed_only/configs/*.json
```

数据：

```text
guagongzhuang_multimodal_asym_clean.zarr
guagongzhuang_huangdong_multimodal_asym_clean.zarr
```

上传服务器后，需要修改 task yaml 里的：

```text
dataset.zarr_path
```

## 13. 当前状态快照

当前已经完成：

```text
新挂工装 raw/zarr 数据识别完成。
新挂工装数据已按无夹爪 6 维 action 清洗完成。
已生成点云版、多模态版、非对称多模态版 clean zarr。
已新增挂工装 asym task yaml。
已实现真实 checkpoint 部署脚本 deploy_real_policy.py。
已修正挂工装可视化语义：固定白色母线 + 随 TCP 运动的工装。
已重新生成挂工装可视化视频和总览视频。
```

需要后续注意：

```text
如果把新挂工装数据上传服务器，务必上传 6 维 clean zarr。
训练新挂工装时务必使用 real_guagongzhuang* task yaml。
部署新挂工装时务必使用 action_mode=delta_tcp_pose。
如果末端工具重新拆装，先重标定 TCP，再考虑部署。
```

## 14. 快速命令索引

### 14.1 重新清洗新挂工装数据

```bash
cd E:\.MP1\real_robot_ur12e_d405_speed_only

python scripts\clean_all_real_datasets.py --root E:\.MP1\real_robot_ur12e_d405_speed_only\data --overwrite --output-action-mode delta_tcp_pose --no-include-gripper-state

python scripts\clean_all_real_datasets_multimodal.py --root E:\.MP1\real_robot_ur12e_d405_speed_only\data --overwrite --output-action-mode delta_tcp_pose --no-include-gripper-state
```

### 14.2 重新生成 asym clean zarr

```bash
python scripts\prepare_multimodal_real_dataset.py --input-zarr E:\.MP1\real_robot_ur12e_d405_speed_only\data\zarr\guagongzhuang\guagongzhuang.zarr --output-zarr E:\.MP1\real_robot_ur12e_d405_speed_only\data\clean_zarr\guagongzhuang_multimodal_asym_clean.zarr --overwrite --global-image-height 128 --global-image-width 128 --wrist-image-height 96 --wrist-image-width 96 --output-action-mode delta_tcp_pose --no-include-gripper-state
```

```bash
python scripts\prepare_multimodal_real_dataset.py --input-zarr E:\.MP1\real_robot_ur12e_d405_speed_only\data\zarr\guagongzhuang_huangdong\guagongzhuang_huangdong.zarr --output-zarr E:\.MP1\real_robot_ur12e_d405_speed_only\data\clean_zarr\guagongzhuang_huangdong_multimodal_asym_clean.zarr --overwrite --global-image-height 128 --global-image-width 128 --wrist-image-height 96 --wrist-image-width 96 --output-action-mode delta_tcp_pose --no-include-gripper-state
```

### 14.3 训练新挂工装

```bash
cd E:\.MP1

bash ./auto_run_real_multimodal_asym.bash 3 pa-pisb real_guagongzhuang_multimodal_asym_clean_upload 0

bash ./auto_run_real_multimodal_asym.bash 3 pa-pisb-ib real_guagongzhuang_huangdong_multimodal_asym_clean_upload 0
```

### 14.4 生成视频

```bash
cd E:\.MP1\real_robot_ur12e_d405_speed_only

python scripts\render_real_episode_showcase.py --episode-dir E:\.MP1\real_robot_ur12e_d405_speed_only\data\raw\guagongzhuang\episode_0031 --output-mp4 "C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略\multimedia\guagongzhuang_episode_0031_showcase.mp4" --mode preserve --layout supp_clean

python scripts\render_real_episode_showcase.py --episode-dir E:\.MP1\real_robot_ur12e_d405_speed_only\data\raw\guagongzhuang_huangdong\episode_0029 --output-mp4 "C:\Users\Administrator\Desktop\论文投稿\pose-estimation-paper\RAL-机械臂策略\multimedia\guagongzhuang_huangdong_episode_0029_showcase.mp4" --mode preserve --layout supp_clean
```

### 14.5 部署评测

```bash
cd E:\.MP1\real_robot_ur12e_d405_speed_only

python scripts\deploy_real_policy.py --checkpoint E:\path\to\latest.ckpt --config configs\collect_pole_pickoff_laptop_remote.json --num-trials 20 --max-episode-s 180 --control-hz 5 --output-dir deploy_results\guagongzhuang
```

如果工具坐标临时需要修正：

```bash
python scripts\deploy_real_policy.py --checkpoint E:\path\to\latest.ckpt --config configs\your_deploy_config.json --action-rotation-correction-rpy-deg 0 0 180 --num-trials 20
```

## 15. 给后续模型的任务执行建议

如果后续模型要继续做代码工作，建议先读：

```text
E:\.MP1\MP1_REAL_ROBOT_FULL_CONTEXT_CN.md
E:\.MP1\REAL_ZARR_TRAINING_UPLOAD.md
E:\.MP1\REAL_DATA_PAPER_TUTORIAL.md
E:\.MP1\MP1_PAPER_RECOMMENDATION_CN.md
```

如果要动清洗流程，先看：

```text
prepare_clean_real_dataset.py
prepare_multimodal_real_dataset.py
```

如果要动视频，先看：

```text
render_real_episode_showcase.py
```

如果要动训练，先看：

```text
train_real.py
mp1/config/*.yaml
mp1/config/task/*.yaml
real_multimodal_dataset.py
```

如果要动部署，先看：

```text
deploy_real_policy.py
real_robot_utils.py
采集 JSON config
```

最重要的约束：

```text
不要把有夹爪任务和无夹爪任务混为一类。
不要把真实成功率当成训练日志自动指标。
不要在未确认 TCP/工具/坐标系一致的情况下直接上真机部署。
不要让目标母线随晃动平台运动，挂工装任务里目标母线是固定的。
```
