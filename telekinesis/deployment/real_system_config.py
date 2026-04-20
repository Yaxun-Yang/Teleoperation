#!/usr/bin/env python3
"""
Configuration file for real system deployment
"""

import os
from path_config import DATA_DIR

# Model configuration
CHECKPOINT_PATH = os.path.join(DATA_DIR, "action2_new_eightyK.ckpt")
DEVICE = "cuda:0"

# PyBullet IK configuration
KINOVA_URDF_PATH = "/home/yzwang/leapUMI/leaphandproject_ws/src/kinova-ros/kinova_description/urdf/robot.urdf"
LEAPHAND_URDF_PATH = "../leap_hand_mesh_right/robot_pybullet.urdf"  # Relative to telekinesis directory
KINOVA_END_EFFECTOR_INDEX = 9
LEAP_END_EFFECTOR_INDICES = [4, 9, 14, 19]

# Azure Kinect camera configuration
CAMERA_FPS = 30
COLOR_RESOLUTION = "1536P"  # 720P or 1080P
DEPTH_MODE = "NFOV_UNBINNED"
SHORTER_SIDE = 720
ZFAR = 2.0

# ROS topics (keeping for compatibility)
JOINT_STATE_TOPIC = "/j2n6s300/joint_states"
LEAPHAND_TOPIC = "/leaphand/joint_states"
LEAPHAND_CMD_TOPIC = "/leaphand/joint_cmd"

# Kinova action server
KINOVA_ACTION_SERVER = "/j2n6s300_driver/joints_action/joint_angles"

# TF frames (keeping for reference)
BASE_FRAME = "base_link"
EE_FRAME = "j2n6s300_end_effector"

# Image processing
IMAGE_SIZE = (92, 92)  # Resize dimensions
CROP_MARGINS = (80, 80)  # Left/right crop margins

# Execution parameters
CONTROL_FREQUENCY = 10  # Hz
ACTION_EXECUTION_DELAY = 0.1  # seconds between actions in a chunk

# Jump detection parameters
ARM_JUMP_THRESHOLD = 0.5  # radians, similar to rollout_policy.py
LEAPHAND_JUMP_THRESHOLD = 0.3  # radians, adjust based on LeapHand characteristics
MAX_JOINT_VELOCITY = 0.2  # radians per step, maximum joint velocity limit

# Safety limits (adjust according to your setup)
POSITION_LIMITS = {
    'x': (-3.0, 3.0),
    'y': (-3.0, 3.0), 
    'z': (0.1, 1.2)
}

JOINT_LIMITS = {
    'joint1': (-3.14, 3.14),
    'joint2': (-3.14, 3.14),
    'joint3': (-3.14, 3.14),
    'joint4': (-3.14, 3.14),
    'joint5': (-3.14, 3.14),
    'joint6': (-3.14, 3.14)
}

# LeapHand limits (16 DOF) - extracted from leap_hand_mesh_right/robot_pybullet.urdf
LEAPHAND_LIMITS = {
    'lower': [
        -0.349,  # joint 0 (finger 1 PIP)
        -0.314,  # joint 1 (finger 1 MCP) 
        -0.506,  # joint 2 (finger 1 DIP)
        -0.366,  # joint 3 (finger 1 fingertip)
        -0.349,  # joint 4 (finger 2 PIP)
        -0.314,  # joint 5 (finger 2 MCP)
        -0.506,  # joint 6 (finger 2 DIP)
        -0.366,  # joint 7 (finger 2 fingertip)
        -0.349,  # joint 8 (finger 3 PIP)
        -0.314,  # joint 9 (finger 3 MCP)
        -0.506,  # joint 10 (finger 3 DIP)
        -0.366,  # joint 11 (finger 3 fingertip)
        -0.349,  # joint 12 (thumb base)
        -0.47,   # joint 13 (thumb PIP)
        -1.20,   # joint 14 (thumb DIP)
        -1.34    # joint 15 (thumb fingertip)
    ],
    'upper': [
        0.349,   # joint 0 (finger 1 PIP)
        2.23,    # joint 1 (finger 1 MCP)
        1.885,   # joint 2 (finger 1 DIP)
        2.042,   # joint 3 (finger 1 fingertip)
        0.349,   # joint 4 (finger 2 PIP)
        2.23,    # joint 5 (finger 2 MCP)
        1.885,   # joint 6 (finger 2 DIP)
        2.042,   # joint 7 (finger 2 fingertip)
        0.349,   # joint 8 (finger 3 PIP)
        2.23,    # joint 9 (finger 3 MCP)
        1.885,   # joint 10 (finger 3 DIP)
        2.042,   # joint 11 (finger 3 fingertip)
        2.094,   # joint 12 (thumb base)
        2.443,   # joint 13 (thumb PIP)
        1.90,    # joint 14 (thumb DIP)
        1.88     # joint 15 (thumb fingertip)
    ]
}
