# QB_Retargeting

项目用于基于手部检测/深度相机对 `xArm7` 机械臂 与 清瑞博源（QB）灵巧手 进行遥操作与重定向（retargeting）。

**简介**
- 本仓库实现了从相机（RGB / 深度）、视频或离线数据对人体手部动作进行检测并映射到机器人手臂与手的控制上（含仿真与真实机器人）。
- 支持实时 teleoperation（手势驱动机械臂+手）、离线位置重定向与 DexPilot 风格的指尖力控模式。

**功能**
- 实时深度相机/摄像头手部检测并重定向到机器人（xArm7 + QB Hand）
- 通过重定向优化器（vector / position / dexpilot）计算关节目标
- 真机控制封装（`real_world_controller.XArm7QB`）支持 xArm7（IP 控制）与 QB 手（串口）
- 支持离线数据处理与视频回放（`retarget_from_video.py`、`retarget_from_dataset.py`）

**仓库结构（关键文件）**
- **主入口/脚本**: `retarget_real_world.py`, `retarget_real_hand.py`, `retarget_from_depth_camera_teleop.py`, `retarget_from_camera_teleop.py`, `retarget_from_video.py`
- **真实机器人驱动**: `real_world_controller.py`（`XArm7QB` 封装 xArm + QB Hand 串口）
- **灵巧手控制**: `qb_hand/dexterous_hand_controller.py`（串口协议、力控示例）
- **配置/重定向**: `config.py`, `yaml_configs/`（teleop / offline 下的手型配置）
- **环境/依赖**: `min_environment.yml`, `min_packages.txt`

**硬件要求**
- xArm7: 能通过局域网访问的 IP （默认示例 `192.168.1.204`，请按实际设备修改）
- QB 灵巧手: 串口（例如 `COM3`）连接，默认波特率 115200
- 可选：RealSense 等深度相机（若使用深度 teleop）

**软件依赖**
- 推荐使用 conda 创建环境，仓库提供 `min_environment.yml` 和 `min_packages.txt`。

示例（conda）：
```bash
conda env create -f min_environment.yml
conda activate chenyc
pip install -r requirements_optional.txt  # 若有额外 pip 依赖，可手动安装
```

如果不使用 conda，可参考 `min_packages.txt` 的 pip 列表安装常用依赖，例如 `pyserial`, `opencv-python`, `pyrealsense2` 等。

**快速开始（仿真/可视化）**
- 在不连接真实机器人时，可以运行示例脚本查看重定向结果（在窗口中可视化）：

```bash
python retarget_from_depth_camera_teleop.py
# 或者
python retarget_from_camera_teleop.py
```

这些脚本会加载默认的重定向配置（参见 `yaml_configs/teleop/`），并打开摄像头/深度流，进行实时检测与仿真展示。

**快速连接真实机器人（危险操作，请先确认安全）**
- 编辑 `real_world_controller.py` 或在构造 `XArm7QB` 时传入正确参数：`arm_ip_address`（xArm IP）、`hand_port`（COM 口）等。
- 运行示例：

```bash
python retarget_real_world.py
```

默认 `retarget_real_world.py` 在脚本底部调用示例：
- `robot_name=RobotName.mimic_qb`，`retargeting_type=RetargetingType.dexpilot`，`hand_type=HandType.left`。

注意：真实机器人操作会移动机械臂/手，请确保环境无障碍且在运行前设置好停止策略与紧急停止（`Ctrl+C`）。

**配置文件说明**
- 所有重定向参数存放在 `yaml_configs/` 下：
  - `teleop/`：实时 teleoperation 配置（vector / dexpilot 等）
  - `offline/`：离线位置重定向配置
- 在代码中可通过 `config.get_default_config_path(robot_name, retargeting_type, hand_type)` 获取默认配置路径。

**主要脚本说明（简要）**
- `retarget_from_depth_camera_teleop.py`: 使用深度相机做实时 teleop（arm + hand 仿真/可视化）
- `retarget_from_camera_teleop.py`: 使用 RGB 摄像头 + MediaPipe 做实时 teleop
- `retarget_real_hand.py` / `retarget_real_world.py`: 将重定向命令发送到真实机器人（xArm7 + QB Hand）
- `qb_hand/dexterous_hand_controller.py`: QB 手串口协议实现，包含角度查询、力控示例与常用预设

**安全与调试建议**
- 先在仿真/可视化模式下验证动作轨迹，再连接真实机器人。  
- 进行真实机器人测试前，先设置合适的初始姿态与较小速度/力参数。  
- `real_world_controller.XArm7QB` 中的默认 `arm_mode` 与 PID 参数可以根据实际设备调整。

**常见问题**
- 如果连接 xArm 失败：检查 IP、网线、xArm 控制器是否启用网络控制；查看 xArm API 返回的错误码。  
- 如果串口连接 QB 手失败：确认 COM 口号、波特率、串口是否被其它程序占用。  
- 运行依赖错误：优先使用提供的 conda 环境，按照 `min_environment.yml` 创建环境。

