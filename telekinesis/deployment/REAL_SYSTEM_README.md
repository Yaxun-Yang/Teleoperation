# Diffusion Policy 实机部署说明

## 概述

本文档介绍如何在实机上部署 Diffusion Policy，控制 Kinova 机械臂（6DOF）和 LeapHand（16DOF）进行操作任务。

## 系统要求

### 硬件要求
- Kinova 机械臂（6DOF）
- LeapHand 灵巧手（16DOF）
- RealSense 相机或其他兼容相机
- 运行 Ubuntu 18.04/20.04 的计算机
- 支持 CUDA 的 GPU（推荐）

### 软件要求
- ROS Noetic
- Python 3.8+
- PyTorch
- MoveIt
- 训练好的 Diffusion Policy 模型

## 安装和配置

### 1. 环境准备

```bash
# 确保 ROS 环境已设置
source /opt/ros/noetic/setup.bash
source /path/to/your/workspace/devel/setup.bash
```

### 2. 依赖包安装

```bash
# 安装 Python 依赖
pip install torch torchvision torchaudio
pip install opencv-python
pip install scipy
pip install hydra-core
pip install dill

# 安装 ROS 依赖
sudo apt-get install ros-noetic-moveit
sudo apt-get install ros-noetic-realsense2-camera
```

### 3. 模型准备

确保您有训练好的 Diffusion Policy 模型检查点文件（.ckpt 格式）。

## 使用方法

### 1. 启动所需的 ROS 节点

在不同的终端中依次启动以下节点：

```bash
# 终端 1: 启动 Kinova 驱动
roslaunch kinova_bringup kinova_robot.launch

# 终端 2: 启动 LeapHand 驱动
# （根据您的 LeapHand 驱动启动相应的 launch 文件）
roslaunch leaphand_driver leaphand.launch

# 终端 3: 启动相机
roslaunch realsense2_camera rs_camera.launch

# 终端 4: 启动 MoveIt
roslaunch kinova_leaphand_moveit move_group.launch
```

### 2. 启动 Diffusion Policy

```bash
# 使用默认参数
./launch_real_system.sh

# 或指定模型路径和设备
./launch_real_system.sh /path/to/your/model.ckpt cuda:0

# 或直接运行 Python 脚本
python3 rollout_policy_real_system.py --checkpoint /path/to/model.ckpt --device cuda:0
```

## 文件说明

### 主要文件

1. **`rollout_policy_real_system.py`** - 主要的部署脚本
   - 加载训练好的 Diffusion Policy 模型
   - 处理传感器数据（相机图像、关节状态）
   - 执行动作预测和机器人控制

2. **`real_system_config.py`** - 配置文件
   - 包含所有系统参数和设置
   - 可根据具体硬件配置进行调整

3. **`launch_real_system.sh`** - 启动脚本
   - 自动检查系统状态
   - 简化启动流程

### 核心功能

#### RealSystemDiffusionPolicy 类

- **`load_policy()`** - 加载预训练的 Diffusion Policy 模型
- **`init_ros_interfaces()`** - 初始化 ROS 发布者、订阅者和服务
- **`prepare_model_input()`** - 准备模型输入数据
- **`predict_action()`** - 使用模型预测动作序列
- **`execute_action_chunk()`** - 执行预测的动作序列
- **`ee_to_joint_commands()`** - 使用 MoveIt 进行逆运动学计算

## 配置参数

### 重要参数（在 `real_system_config.py` 中）

```python
# 模型配置
CHECKPOINT_PATH = "/media/yaxun/manipulation1/leaphandproject_ws/checkpoints/latest.ckpt"
DEVICE = "cuda:0"

# 控制频率
CONTROL_FREQUENCY = 10  # Hz

# 图像处理
IMAGE_SIZE = (92, 92)
CROP_MARGINS = (80, 80)

# 安全限制
POSITION_LIMITS = {
    'x': (-1.0, 1.0),
    'y': (-1.0, 1.0), 
    'z': (0.1, 1.2)
}
```

## 动作空间

### 输入观测
- **相机图像**: RGB 图像 (92x92)
- **末端执行器位置**: 3D 位置 (x, y, z)
- **末端执行器姿态**: 四元数表示
- **LeapHand 关节位置**: 16 个关节的角度

### 输出动作
- **末端执行器目标位置**: 3D 坐标 (x, y, z)
- **末端执行器目标姿态**: 6D 旋转表示
- **LeapHand 目标关节角度**: 16 个关节的目标角度

总动作维度: 3 + 6 + 16 = 25

## 故障排除

### 常见问题

1. **"No action predicted"**
   - 检查传感器数据是否正常接收
   - 确认模型文件路径正确
   - 检查 GPU 内存是否充足

2. **"MoveIt planning failed"**
   - 检查目标位置是否在工作空间内
   - 确认 MoveIt 配置正确
   - 检查关节限制设置

3. **"ROS topic not found"**
   - 确认所有驱动节点正常运行
   - 检查 topic 名称是否正确

### 调试建议

1. 启用详细日志输出
2. 使用 `rostopic echo` 检查数据流
3. 使用 RViz 可视化机器人状态
4. 逐步测试各个模块功能

## 安全注意事项

1. **首次运行前务必确保**：
   - 机器人工作空间内无障碍物
   - 急停按钮随时可达
   - 有经验的操作员在场

2. **运行期间**：
   - 监控机器人动作是否异常
   - 注意安全限制是否生效
   - 准备随时终止程序

3. **参数调整**：
   - 从较低的速度和精度开始
   - 逐步增加控制频率
   - 仔细调整安全限制

## 性能优化

1. **GPU 使用**：
   - 确保 CUDA 可用
   - 调整批处理大小
   - 考虑使用 TensorRT 加速

2. **实时性**：
   - 优化图像预处理
   - 调整控制频率
   - 使用异步执行

## 扩展和定制

本框架支持以下定制：

1. **不同的机器人配置**
2. **其他类型的传感器**
3. **不同的控制策略**
4. **自定义的安全检查**

根据具体需求修改相应的配置和代码即可。

## 联系和支持

如有问题或需要技术支持，请联系开发团队。
