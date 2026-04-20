#!/usr/bin/env python3
"""
Path configuration for deployment directory
部署目录的路径配置
"""

import os

# 获取当前文件的目录
DEPLOYMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 计算其他重要目录的路径
TELEKINESIS_DIR = os.path.dirname(DEPLOYMENT_DIR)
SRC_DIR = os.path.dirname(TELEKINESIS_DIR)
WORKSPACE_DIR = os.path.dirname(SRC_DIR)

# 数据目录
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")

# diffusion_policy 目录
DIFFUSION_POLICY_DIR = os.path.join(SRC_DIR, "diffusion_policy")

# 打印路径信息（用于调试）
def print_paths():
    print("=== 路径配置 ===")
    print(f"部署目录: {DEPLOYMENT_DIR}")
    print(f"telekinesis 目录: {TELEKINESIS_DIR}")
    print(f"src 目录: {SRC_DIR}")
    print(f"工作空间目录: {WORKSPACE_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"diffusion_policy 目录: {DIFFUSION_POLICY_DIR}")

if __name__ == "__main__":
    print_paths()
