#!/usr/bin/env python3
"""
Real-time deployment of DP3 (3D Diffusion Policy) on real robot system
Robot: Kinova arm (6DOF) + LeapHand (16DOF)
Network output: position(3) + quaternion(4) + hand_qpos(16) = 23 dimensions
Network input: agent_pos(23) + point_cloud (generated from cam_K.txt)
"""

import os
import sys
import rospy
import numpy as np
import torch
import hydra
import dill
import time
import cv2
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation
import pybullet as p
import actionlib
import kinova_msgs.msg
from kinova_msgs.msg import JointTorque
import pyk4a
from pyk4a import Config, PyK4A

# Add DP3 and OmegaConf imports
from diffusion_policy_3d.policy.dp3 import DP3
from omegaconf import OmegaConf, DictConfig

# Add necessary paths
import path_config
sys.path.append(path_config.DIFFUSION_POLICY_DIR)
sys.path.append(path_config.SRC_DIR)

# Add DP3 paths - for 3D diffusion policy
dp3_path = os.path.join(path_config.SRC_DIR, 'telekinesis', 'deployment', '3D-Diffusion-Policy', '3D-Diffusion-Policy')
if os.path.exists(dp3_path):
    sys.path.append(dp3_path)

# Import modules
from scipy.spatial.transform import Rotation
import real_system_config as config
from kinova_node import KinovaNode

# Add LeapHand utilities
sys.path.append(os.path.join(path_config.SRC_DIR, 'leap_hand_utils'))
from leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils.leap_hand_utils as lhu


def filter_outliers(points, method='iqr', iqr_factor=1.5, zscore_threshold=3.0):
    """
    过滤点云中的异常值
    
    Args:
        points: 3D点坐标 (N, 3)
        method: 过滤方法 'iqr' 或 'zscore'
        iqr_factor: IQR方法的倍数因子
        zscore_threshold: Z-score方法的阈值
    
    Returns:
        filtered_points: 过滤后的点
        outlier_mask: 异常值的布尔掩码
    """
    if len(points) == 0:
        return points, np.array([], dtype=bool)
    
    outlier_mask = np.zeros(len(points), dtype=bool)
    
    if method == 'iqr':
        # 使用四分位距（IQR）方法检测异常值
        for dim in range(3):
            q1 = np.percentile(points[:, dim], 25)
            q3 = np.percentile(points[:, dim], 75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_factor * iqr
            upper_bound = q3 + iqr_factor * iqr
            
            dim_outliers = (points[:, dim] < lower_bound) | (points[:, dim] > upper_bound)
            outlier_mask |= dim_outliers
            
    elif method == 'zscore':
        # 使用Z-score方法检测异常值
        for dim in range(3):
            mean = np.mean(points[:, dim])
            std = np.std(points[:, dim])
            if std > 0:
                z_scores = np.abs((points[:, dim] - mean) / std)
                dim_outliers = z_scores > zscore_threshold
                outlier_mask |= dim_outliers
    
    # 返回非异常值点和异常值掩码
    filtered_points = points[~outlier_mask]
    
    return filtered_points, outlier_mask


def spatial_grid_sampling(points, target_points):
    """
    空间网格采样：将3D空间划分为网格，每个网格采样固定数量的点
    添加异常值过滤来改善边界计算
    """
    if len(points) == 0:
        return np.array([], dtype=int)
    
    # 过滤异常值来计算更准确的边界
    filtered_points, outlier_mask = filter_outliers(points)
    
    # 使用过滤后的点计算边界
    if len(filtered_points) > 0:
        min_coords = filtered_points.min(axis=0)
        max_coords = filtered_points.max(axis=0)
        print(f"Boundary calculation: used {len(filtered_points)}/{len(points)} points (filtered {np.sum(outlier_mask)} outliers)")
    else:
        # 如果所有点都被认为是异常值，回退到原始方法
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        print(f"Warning: All points considered outliers, using original boundary calculation")
    
    # 计算网格大小（自适应）
    ranges = max_coords - min_coords
    print(f"Boundary ranges: X={ranges[0]:.3f}, Y={ranges[1]:.3f}, Z={ranges[2]:.3f}")
    
    # 根据目标点数自适应网格数量
    grid_size = max(1, int(np.cbrt(len(points) / target_points) * 2))
    print(f"Grid size: {grid_size}³ = {grid_size**3} cells")
    
    # 为每个维度创建网格
    grid_indices = []
    for i in range(3):
        if ranges[i] > 0:
            # 使用过滤后的边界，但对所有点进行网格分配
            grid_indices.append(((points[:, i] - min_coords[i]) / ranges[i] * grid_size).astype(int))
            # 确保网格索引在有效范围内
            grid_indices[i] = np.clip(grid_indices[i], 0, grid_size - 1)
        else:
            grid_indices.append(np.zeros(len(points), dtype=int))
    
    # 组合网格索引
    grid_coords = np.column_stack(grid_indices)
    
    # 为每个网格单元选择点
    unique_grids, inverse_indices = np.unique(grid_coords, axis=0, return_inverse=True)
    selected_indices = []
    
    points_per_grid = max(1, target_points // len(unique_grids))
    print(f"Target points per grid: {points_per_grid}, Total grids with points: {len(unique_grids)}")
    
    for grid_idx in range(len(unique_grids)):
        grid_mask = inverse_indices == grid_idx
        grid_point_indices = np.where(grid_mask)[0]
        
        if len(grid_point_indices) > 0:
            # 在该网格中随机选择点
            n_select = min(points_per_grid, len(grid_point_indices))
            selected = np.random.choice(grid_point_indices, n_select, replace=False)
            selected_indices.extend(selected)
    
    # 如果选择的点不够，随机补充
    if len(selected_indices) < target_points:
        remaining = target_points - len(selected_indices)
        print(f"Remaining points needed: {remaining}")
        all_indices = set(range(len(points)))
        available = list(all_indices - set(selected_indices))
        if available:
            additional = np.random.choice(available, min(remaining, len(available)), replace=False)
            selected_indices.extend(additional)
    
    return np.array(selected_indices[:target_points])


def uniform_pointcloud_sampling(points, colors, target_points=2048):
    """
    使用空间网格采样进行均匀点云采样
    
    Args:
        points: 原始3D点坐标 (N, 3)
        colors: 原始颜色 (N, 3)
        target_points: 目标点数
    
    Returns:
        sampled_points: 采样后的点坐标
        sampled_colors: 采样后的颜色
    """
    n_points = len(points)
    if n_points <= target_points:
        # 如果点数不足，进行上采样
        indices = np.random.choice(n_points, target_points, replace=True)
        return points[indices], colors[indices]
    
    print(f"Applying spatial grid sampling to {n_points} points -> {target_points} points")
    
    # 只使用空间网格采样
    grid_indices = spatial_grid_sampling(points, target_points)
    
    print(f"Spatial grid sampling selected {len(grid_indices)} points")
    
    return points[grid_indices], colors[grid_indices]


class LeapNode:
    """LeapHand control node - with enhanced error handling"""
    def __init__(self):
        # Some parameters
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(
            np.zeros(16))

        # Motor IDs for LeapHand
        self.motors = motors = [0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        
        self.dxl_client = None  # Initialize to None first
        
        # Try to connect to LeapHand on different ports
        ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "COM13"]
        for port in ports:
            try:
                print(f"Trying to connect to LeapHand on {port}...")
                self.dxl_client = DynamixelClient(motors, port, 4000000)
                self.dxl_client.connect()
                print(f"Successfully connected to LeapHand on {port}")
                break
            except Exception as e:
                print(f"Failed to connect to {port}: {e}")
                self.dxl_client = None
        
        if self.dxl_client is None:
            print("WARNING: Could not connect to LeapHand - running in simulation mode")
            return
            
        # Configure LeapHand if connection succeeded
        try:
            self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
            self.dxl_client.set_torque_enabled(motors, True)
            self.dxl_client.sync_write(
                motors, np.ones(len(motors)) * self.kP, 84, 2)
            self.dxl_client.sync_write(
                [0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
            self.dxl_client.sync_write(
                motors, np.ones(len(motors)) * self.kI, 82, 2)
            self.dxl_client.sync_write(
                motors, np.ones(len(motors)) * self.kD, 80, 2)
            self.dxl_client.sync_write(
                [0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)
            self.dxl_client.sync_write(motors, np.ones(
                len(motors)) * self.curr_lim, 102, 2)
            self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
            print("LeapHand initialized successfully")
        except Exception as e:
            print(f"Error configuring LeapHand: {e}")
            self.dxl_client = None

    def set_leap(self, pose):
        if self.dxl_client is None:
            return  # Skip if not connected
        try:
            self.prev_pos = self.curr_pos
            self.curr_pos = np.array(pose)
            self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        except Exception as e:
            print(f"Error setting LeapHand pose: {e}")

    def set_allegro(self, pose):
        if self.dxl_client is None:
            return  # Skip if not connected
        try:
            pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
            self.set_leap(pose)
        except Exception as e:
            print(f"Error setting Allegro pose: {e}")

    def read_pos(self):
        if self.dxl_client is None:
            return np.zeros(16)  # Return dummy data if not connected
        try:
            return self.dxl_client.read_pos()
        except Exception as e:
            print(f"Error reading LeapHand position: {e}")
            return np.zeros(16)

    def shutdown(self):
        if self.dxl_client is not None:
            try:
                self.dxl_client.disconnect()
            except Exception as e:
                print(f"Error disconnecting LeapHand: {e}")


class RealSystemDP3Policy:
    """Real-time deployment of DP3 (3D Diffusion Policy) on real robot system"""
    
    def __init__(self, checkpoint_path, device='cuda:0'):
        """
        Initialize the DP3 policy deployment system
        
        Args:
            checkpoint_path: Path to trained DP3 checkpoint
            device: Device to run the model on
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        
        # Initialize timing stats
        self.timing_stats = {
            'start_time': time.time(),
            'total_iterations': 0,
            'total_inference_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Initialize ROS
        rospy.init_node('dp3_policy_deployment', anonymous=True)
        rospy.loginfo("DP3 Policy Deployment System Starting...")
        
        # Load camera intrinsics from cam_K.txt
        self.load_camera_intrinsics()
        
        # Initialize robot components
        self.init_robot_components()

        # Initialize PyBullet for IK and state sync
        self.init_pybullet()
        
        # Initialize Azure Kinect camera
        self.init_azure_kinect()
        
        # Load DP3 policy
        self.load_dp3_policy(checkpoint_path)
        
        # Policy state
        self.obs_deque = deque(maxlen=self.n_obs_steps)
        self.step_count = 0
        
        # Current robot states
        self.current_joint_states = None
        self.current_leaphand_states = None
        
        # Jump detection thresholds
        self.arm_jump_threshold = config.ARM_JUMP_THRESHOLD
        
        # Action execution settings
        self.pybullet_sync_enabled = True
        
        # Rotation transformers for legacy action formats
        try:
            from diffusion_policy.model.common.rotation_transformer import RotationTransformer
            self.six_to_quat = RotationTransformer('rotation_6d', 'quaternion')
        except ImportError:
            print("Warning: Could not import rotation transformer")
            self.six_to_quat = None
        
        rospy.loginfo("DP3 Policy Deployment System Initialized")
    
    def load_camera_intrinsics(self):
        """Load camera intrinsics from cam_K.txt"""
        try:
            cam_k_path = os.path.join(os.path.dirname(__file__), 'cam_K.txt')
            self.cam_K = np.loadtxt(cam_k_path)
            rospy.loginfo(f"Loaded camera intrinsics from {cam_k_path}")
            print(f"Camera matrix:\n{self.cam_K}")
        except Exception as e:
            rospy.logerr(f"Failed to load camera intrinsics: {e}")
            # Fallback to default Azure Kinect values
            self.cam_K = np.array([
                [607.0, 0.0, 320.0],
                [0.0, 607.0, 240.0], 
                [0.0, 0.0, 1.0]
            ])
            rospy.logwarn("Using default camera intrinsics")
    
    def init_robot_components(self):
        """Initialize robot arm and hand components"""
        try:
            # Initialize Kinova arm
            self.kinova_node = KinovaNode()
            rospy.loginfo("Kinova arm initialized")
            
            # Initialize LeapHand
            self.leap_node = LeapNode()
            rospy.loginfo("LeapHand initialized")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize robot components: {e}")
            raise
    
    def init_pybullet(self):
        """Initialize PyBullet simulation for IK and state synchronization"""
        try:
            p.connect(p.DIRECT)
            # Load robot URDF
            self.kinova_id = p.loadURDF(
                config.KINOVA_URDF_PATH,
                [0.0, 0.0, 0.0],
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
            )

            self.num_joints = p.getNumJoints(self.kinova_id)

            # Initialize joints according to real robot's current position
            try:
                current_joint_angles = self.kinova_node.get_current_joint_angles()
                
                if current_joint_angles is not None and len(current_joint_angles) == 6:
                    print("Initializing PyBullet with real robot joint positions:")
                    for i, joint_angle in enumerate(current_joint_angles):
                        p.resetJointState(self.kinova_id, i + 2, joint_angle)  # joints 2-7 for 6DOF arm
                        print(f"  Joint {i+1}: {joint_angle:.4f} rad")
                    self.current_joint_states = current_joint_angles
                else:
                    print("Could not get real robot joint positions, using default configuration")
                    for i in range(2, 8):
                        p.resetJointState(self.kinova_id, i, 0.0)
            except Exception as e:
                print(f"Error getting real robot joint positions: {e}")
                print("Using default joint configuration")
                for i in range(2, 8):
                    p.resetJointState(self.kinova_id, i, 0.0)
            
            p.setGravity(0, 0, 0)
            p.setRealTimeSimulation(0)
            
            rospy.loginfo("PyBullet initialized for IK and state sync")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize PyBullet: {e}")
            raise
    
    def init_azure_kinect(self):
        """Initialize Azure Kinect camera"""
        try:
            # Configure Azure Kinect
            color_resolution = getattr(pyk4a.ColorResolution, f"RES_{config.COLOR_RESOLUTION}")
            depth_mode = getattr(pyk4a.DepthMode, config.DEPTH_MODE)
            camera_fps = getattr(pyk4a.FPS, f"FPS_{config.CAMERA_FPS}")
            
            self.k4a = PyK4A(
                Config(
                    color_resolution=color_resolution,
                    depth_mode=depth_mode,
                    camera_fps=camera_fps,
                    synchronized_images_only=True,
                )
            )
            self.k4a.start()
            
            # Use cam_K.txt intrinsics instead of Azure Kinect calibration
            # Calculate downscale factor from camera parameters
            capture = self.k4a.get_capture()
            H, W = capture.color.shape[:2]
            self.downscale = config.SHORTER_SIDE / min(H, W)
            self.target_H = int(H * self.downscale)
            self.target_W = int(W * self.downscale)
            
            # Use cam_K.txt and apply downscaling
            self.K = self.cam_K.copy()
            self.K[:2] *= self.downscale
            
            rospy.loginfo(f"Azure Kinect initialized: {W}x{H} -> {self.target_W}x{self.target_H}")
            print(f"Using cam_K.txt intrinsics with downscale factor {self.downscale:.3f}:")
            print(f"Camera matrix after downscale:\n{self.K}")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize Azure Kinect: {e}")
            self.k4a = None
    
    def load_dp3_policy(self, checkpoint_path):
        """Load DP3 policy from checkpoint"""
        print(f"Loading DP3 policy from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check structure
        if 'cfg' not in checkpoint:
            raise ValueError("Checkpoint must contain 'cfg' key")
            
        cfg = checkpoint['cfg']
        print(f"Config type: {type(cfg)}")
        
        # Get policy config
        if hasattr(cfg, 'policy'):
            policy_cfg = cfg.policy
        elif isinstance(cfg, dict) and 'policy' in cfg:
            policy_cfg = cfg['policy']
        else:
            raise ValueError("No policy config found in checkpoint")
            
        print(f"Policy config keys: {list(policy_cfg.keys()) if hasattr(policy_cfg, 'keys') else 'No keys'}")
        
        # Handle shape_meta - first try to get from policy_cfg, then from cfg
        shape_meta = None
        if hasattr(policy_cfg, 'shape_meta'):
            shape_meta = policy_cfg.shape_meta
        elif hasattr(cfg, 'shape_meta'):
            shape_meta = cfg.shape_meta
        elif hasattr(cfg, 'obs_shape_meta'):
            # Build shape_meta from obs_shape_meta if available
            obs_shape_meta = cfg.obs_shape_meta
            shape_meta = {
                'obs': obs_shape_meta,
                'action': {'shape': [23]}  # Our output action dimension
            }
        else:
            # Manual shape_meta as fallback - updated for RGB point clouds
            shape_meta = {
                'obs': {
                    'point_cloud': {'shape': [2048, 6], 'type': 'point_cloud'},  # Changed from 3 to 6 for RGB
                    'agent_pos': {'shape': [23], 'type': 'low_dim'}
                },
                'action': {'shape': [23]}
            }
            
        print(f"Shape meta: {shape_meta}")
        
        # Convert shape_meta to proper format if needed
        if isinstance(shape_meta, (dict, DictConfig)):
            # Use safe_dict_apply to avoid list recursion issues
            def extract_shape(x):
                if isinstance(x, (dict, DictConfig)) and 'shape' in x:
                    return tuple(x['shape'])
                return x
                
            def safe_dict_apply(func, d):
                """Apply function to dict values, avoiding recursion on lists"""
                if isinstance(d, (dict, DictConfig)):
                    result = {}
                    for k, v in d.items():
                        if isinstance(v, (dict, DictConfig)):
                            result[k] = safe_dict_apply(func, v)
                        else:
                            result[k] = func(v)
                    return result
                else:
                    return func(d)
                    
            obs_dict = safe_dict_apply(extract_shape, shape_meta['obs'])
            print(f"Extracted obs_dict: {obs_dict}")
        else:
            raise ValueError(f"Invalid shape_meta type: {type(shape_meta)}")
        
        # Get noise scheduler config and create actual scheduler
        noise_scheduler_cfg = None
        if hasattr(policy_cfg, 'noise_scheduler'):
            noise_scheduler_cfg = policy_cfg.noise_scheduler
        else:
            # Default DDIM scheduler
            noise_scheduler_cfg = {
                '_target_': 'diffusers.schedulers.scheduling_ddim.DDIMScheduler',
                'num_train_timesteps': 100,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'squaredcos_cap_v2',
                'clip_sample': True,
                'set_alpha_to_one': True,
                'steps_offset': 0,
                'prediction_type': 'sample'
            }
        
        # Create actual scheduler object
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=noise_scheduler_cfg['num_train_timesteps'],
            beta_start=noise_scheduler_cfg['beta_start'],
            beta_end=noise_scheduler_cfg['beta_end'],
            beta_schedule=noise_scheduler_cfg['beta_schedule'],
            clip_sample=noise_scheduler_cfg['clip_sample'],
            set_alpha_to_one=noise_scheduler_cfg['set_alpha_to_one'],
            steps_offset=noise_scheduler_cfg['steps_offset'],
            prediction_type=noise_scheduler_cfg['prediction_type']
        )
        
        # Get pointcloud encoder config
        pointcloud_encoder_cfg = None
        if hasattr(policy_cfg, 'pointcloud_encoder_cfg'):
            pointcloud_encoder_cfg = policy_cfg.pointcloud_encoder_cfg
            # Handle OmegaConf interpolation for out_channels
            if isinstance(pointcloud_encoder_cfg, DictConfig):
                pointcloud_encoder_cfg = OmegaConf.to_container(pointcloud_encoder_cfg, resolve=True)
        else:
            # Default pointcloud encoder config
            pointcloud_encoder_cfg = {
                'in_channels': 3,
                'out_channels': 64,
                'use_layernorm': True,
                'final_norm': 'layernorm',
                'normal_channel': False
            }
        
        print(f"Pointcloud encoder config: {pointcloud_encoder_cfg}")
        
        # Create DP3 policy instance
        try:
            self.policy = DP3(
                shape_meta=shape_meta,  # Pass the original shape_meta dict
                noise_scheduler=noise_scheduler,  # Use actual scheduler object
                horizon=getattr(policy_cfg, 'horizon', 16),
                n_action_steps=getattr(policy_cfg, 'n_action_steps', 8),
                n_obs_steps=getattr(policy_cfg, 'n_obs_steps', 2),
                num_inference_steps=getattr(policy_cfg, 'num_inference_steps', 10),
                obs_as_global_cond=getattr(policy_cfg, 'obs_as_global_cond', True),
                crop_shape=getattr(policy_cfg, 'crop_shape', [80, 80]),
                encoder_output_dim=getattr(policy_cfg, 'encoder_output_dim', 64),
                pointcloud_encoder_cfg=OmegaConf.create(pointcloud_encoder_cfg),
                # Add missing parameters from checkpoint
                diffusion_step_embed_dim=getattr(policy_cfg, 'diffusion_step_embed_dim', 128),
                down_dims=getattr(policy_cfg, 'down_dims', [512, 1024, 2048]),
                kernel_size=getattr(policy_cfg, 'kernel_size', 5),
                n_groups=getattr(policy_cfg, 'n_groups', 8),
                condition_type=getattr(policy_cfg, 'condition_type', 'film'),
                use_down_condition=getattr(policy_cfg, 'use_down_condition', True),
                use_mid_condition=getattr(policy_cfg, 'use_mid_condition', True),
                use_up_condition=getattr(policy_cfg, 'use_up_condition', True),
                use_pc_color=getattr(policy_cfg, 'use_pc_color', False),
                pointnet_type=getattr(policy_cfg, 'pointnet_type', 'pointnet')
            )
            
            print("DP3 policy created successfully")
            
            # Store configuration values
            self.horizon = getattr(policy_cfg, 'horizon', 16)
            self.n_obs_steps = getattr(policy_cfg, 'n_obs_steps', 2)
            self.n_action_steps = getattr(policy_cfg, 'n_action_steps', 8)
            
            # Load state dict if available
            if 'state_dicts' in checkpoint and 'ema' in checkpoint['state_dicts']:
                state_dict = checkpoint['state_dicts']['ema']
                self.policy.load_state_dict(state_dict)
                print("Loaded EMA state dict")
            elif 'state_dicts' in checkpoint and 'model' in checkpoint['state_dicts']:
                state_dict = checkpoint['state_dicts']['model']
                self.policy.load_state_dict(state_dict)
                print("Loaded model state dict")
            else:
                print("Warning: No state dict found in checkpoint")
            
            # Move to device and set to eval mode
            self.policy.to(self.device)
            self.policy.eval()
            
            # Network output dimensions: position(3) + quaternion(4) + hand_qpos(16) = 23
            self.action_dim = 23
            self.pos_dim = 3
            self.quat_dim = 4
            self.hand_dim = 16
            
            rospy.loginfo(f"DP3 Policy loaded from {checkpoint_path}")
            rospy.loginfo(f"Config: horizon={self.horizon}, n_obs_steps={self.n_obs_steps}, n_action_steps={self.n_action_steps}")
            
        except Exception as e:
            print(f"Error creating DP3 policy: {e}")
            import traceback
            traceback.print_exc()
            raise
            raise
    
    def get_camera_data(self):
        """Get RGB-D data from Azure Kinect camera"""
        if self.k4a is None:
            return None, None
        
        try:
            capture = self.k4a.get_capture(timeout=33)
            
            # Get color image
            color = capture.color[..., :3].astype(np.uint8)
            print("color image shape: ", color.shape)
            color = cv2.resize(color, (self.target_W, self.target_H), interpolation=cv2.INTER_NEAREST)
            
            # Get depth image
            depth = capture.transformed_depth.astype(np.float32) / 1e3
            print("origin depth shape: ", depth.shape)

            depth = cv2.resize(depth, (self.target_W, self.target_H), interpolation=cv2.INTER_NEAREST)
            depth[(depth < 0.01) | (depth >= config.ZFAR)] = 0

            color = cv2.resize(color, config.IMAGE_SIZE)
            depth = cv2.resize(depth, config.IMAGE_SIZE)
            return color, depth
            
        except Exception as e:
            rospy.logerr(f"Error getting camera data: {e}")
            return None, None

    def generate_point_cloud(self, color_image, depth_image):
        """Generate point cloud from RGB-D images using cam_K.txt intrinsics with advanced sampling"""
        if color_image is None or depth_image is None:
            return None

        try:
            # Use downscaled camera intrinsics
            fx, fy = self.K[0, 0] * 92.0 / 1536, self.K[1, 1] * 92.0 / 2048
            cx, cy = self.K[0, 2] * 92.0 / 1536, self.K[1, 2] * 92.0 / 2048

            # Generate point cloud
            height, width = depth_image.shape
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # Convert to 3D coordinates
            z = depth_image
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            # Filter valid points
            valid_mask = (z > 0.1) & (z < 2.0)  # 10cm to 2m range
            
            # Get XYZ coordinates for valid points
            points_3d = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)
            
            # Get RGB colors for valid points (normalize to [0,1] range)
            colors_rgb = color_image[valid_mask] / 255.0  # Convert from [0,255] to [0,1]
            
            # Target number of points for DP3
            target_points = 2048
            
            # Use advanced point cloud sampling if we have enough points
            if len(points_3d) > target_points:
                print(f"Original point cloud: {len(points_3d)} points")
                
                # Apply uniform spatial grid sampling
                sampled_points, sampled_colors = uniform_pointcloud_sampling(
                    points_3d, colors_rgb, target_points=target_points
                )
                
                print(f"After sampling: {len(sampled_points)} points")
                
                # Combine XYZ and RGB to create 6D point cloud (N, 6)
                points_6d = np.concatenate([sampled_points, sampled_colors], axis=1)
                
            elif len(points_3d) < target_points:
                print(f"Insufficient points ({len(points_3d)}), upsampling to {target_points}")
                
                # Use uniform sampling to upsample
                sampled_points, sampled_colors = uniform_pointcloud_sampling(
                    points_3d, colors_rgb, target_points=target_points
                )
                
                # Combine XYZ and RGB to create 6D point cloud (N, 6)
                points_6d = np.concatenate([sampled_points, sampled_colors], axis=1)
                
            else:
                # Exact number of points
                print(f"Exact target points: {len(points_3d)}")
                points_6d = np.concatenate([points_3d, colors_rgb], axis=1)
            
            # Ensure we have exactly the target number of points
            if len(points_6d) != target_points:
                print(f"Warning: Final point cloud has {len(points_6d)} points instead of {target_points}")
                if len(points_6d) < target_points:
                    # Pad with zeros if still not够 points (6 channels: xyz+rgb)
                    padding_points = np.zeros((target_points - len(points_6d), 6))
                    points_6d = np.vstack([points_6d, padding_points])
                else:
                    # Truncate if too many points
                    points_6d = points_6d[:target_points]
            
            print(f"Final point cloud shape: {points_6d.shape}")
            return points_6d.astype(np.float32)
            
        except Exception as e:
            rospy.logerr(f"Error generating point cloud: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_agent_pos_observation(self):
        """Get agent position observation (23 dimensions)"""
        try:
            # Get current joint states
            arm_joints = self.current_joint_states  # 6 DOF
            hand_joints = self.current_leaphand_states  # 16 DOF
            
            if arm_joints is None or hand_joints is None:
                return None
            
            # Get end-effector pose from PyBullet
            ee_pos, ee_rot = self.get_ee_pose_from_pybullet()
            if ee_pos is None or ee_rot is None:
                return None
            
            # Convert rotation matrix to quaternion
            ee_quat = Rotation.from_matrix(ee_rot).as_quat()  # [x, y, z, w]
            
            # Create 23-dimensional agent position: position(3) + quaternion(4) + hand_qpos(16) = 23
            agent_pos = np.concatenate([
                ee_pos,                  # 3 - position
                ee_quat,                 # 4 - quaternion
                hand_joints              # 16 - hand joint positions
            ])
            
            return agent_pos.astype(np.float32)
            
        except Exception as e:
            rospy.logerr(f"Error getting agent pos observation: {e}")
            return None
    
    def get_ee_pose_from_pybullet(self):
        """Get end-effector pose from PyBullet simulation"""
        try:
            link_state = p.getLinkState(self.kinova_id, config.KINOVA_END_EFFECTOR_INDEX)
            ee_pos = np.array(link_state[4])  # Position
            ee_quat = np.array(link_state[5])  # Quaternion (x, y, z, w)
            
            # Convert quaternion to rotation matrix
            ee_rot = Rotation.from_quat(ee_quat).as_matrix()
            
            return ee_pos, ee_rot
        except Exception as e:
            rospy.logerr(f"Error getting EE pose from PyBullet: {e}")
            return None, None
    
    def ee_to_joint_commands(self, target_pos, target_quat):
        """
        Convert end-effector pose to joint commands using PyBullet IK
        """
        try:
            # Convert scipy quaternion (x,y,z,w) to PyBullet quaternion format if needed
            if len(target_quat) == 4:
                # Assume it's in (x,y,z,w) format, convert to PyBullet format
                pybullet_quat = [target_quat[0], target_quat[1], target_quat[2], target_quat[3]]
            else:
                rospy.logerr(f"Invalid quaternion format: {target_quat}")
                return None
            
            # Use PyBullet IK to calculate joint positions
            joint_positions = p.calculateInverseKinematics(
                self.kinova_id,
                config.KINOVA_END_EFFECTOR_INDEX,
                target_pos,
                pybullet_quat,
                maxNumIterations=50,
                residualThreshold=0.0001
            )
            print("ik joints: ", joint_positions)
            # Return only the first 6 joint positions (arm joints)
            if len(joint_positions) >= 6:
                arm_joint_positions = np.array(joint_positions[:6])  # Skip first 2 joints (base)
                return arm_joint_positions
            else:
                rospy.logwarn("PyBullet IK returned insufficient joint positions")
                return None
                
        except Exception as e:
            rospy.logerr(f"PyBullet IK failed: {e}")
            return None
    
    def execute_arm_joint_command(self, joint_positions):
        """Execute joint command on Kinova arm using Kinova node"""
        try:
            print("arm joint positions: ", joint_positions)
            success = self.kinova_node.move_to_joint_angles(joint_positions, blocking=True)
            return success
            
        except Exception as e:
            rospy.logerr(f"Failed to execute arm command: {e}")
            return False
    
    def update_pybullet_arm_qpos(self, joint_positions):
        """
        Update PyBullet simulation with real robot arm joint positions
        
        Args:
            joint_positions: Current joint positions from real robot (6 DOF array)
        """
        try:
            if joint_positions is None or len(joint_positions) != 6:
                rospy.logwarn(f"Invalid joint positions for PyBullet update: {joint_positions}")
                return False
            
            # Update PyBullet joint states (joints 2-7 correspond to the 6 arm joints)
            for i, joint_pos in enumerate(joint_positions):
                p.resetJointState(self.kinova_id, i + 2, joint_pos)
            
            # Step simulation to update kinematics
            p.stepSimulation()
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Error updating PyBullet arm qpos: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel any ongoing arm actions
            if hasattr(self, 'kinova_node'):
                self.kinova_node.cancel_all_goals()
            
            # Stop Azure Kinect
            if self.k4a is not None:
                self.k4a.stop()
            
            # Disconnect PyBullet
            p.disconnect()
            
            rospy.loginfo("Cleanup completed")
        except Exception as e:
            rospy.logerr(f"Error during cleanup: {e}")
    
    def get_observation(self):
        """Get current observation for DP3 policy"""
        try:
            # Get RGB-D data
            color_image, depth_image = self.get_camera_data()
            if color_image is None or depth_image is None:
                return None
            
            # Generate point cloud
            point_cloud = self.generate_point_cloud(color_image, depth_image)
            if point_cloud is None:
                return None
            
            # Get agent position (23 dimensions)
            agent_pos = self.get_agent_pos_observation()
            if agent_pos is None:
                return None
            
            # Create observation dictionary for DP3
            obs = {
                'point_cloud': torch.tensor(point_cloud).float().to(self.device),
                'agent_pos': torch.tensor(agent_pos).float().to(self.device)
            }
            
            return obs
            
        except Exception as e:
            rospy.logerr(f"Error getting observation: {e}")
            return None
    

    def prepare_model_input(self):
        """Prepare input for DP3 policy: agent_pos(23) + point_cloud"""
        # Get current observation
        current_obs = self.get_observation()
        if current_obs is None:
            return None
        
        # Add current observation to deque
        self.obs_deque.append(current_obs)
        
        # If we don't have enough observations yet, duplicate current observation
        while len(self.obs_deque) < self.n_obs_steps:
            self.obs_deque.append(current_obs)
        
        # Stack observations
        batch_obs = {}
        for key in self.obs_deque[0].keys():
            batch_obs[key] = torch.stack([obs[key] for obs in self.obs_deque])
        
        # Add batch dimension
        model_input = {key: value[None].to(self.device) for key, value in batch_obs.items()}
        
        return model_input
    
    def predict_action(self):
        """Predict action using diffusion policy"""
        model_input = self.prepare_model_input()
        if model_input is None:
            return None, 0.0

        with torch.no_grad():
            start_time = time.perf_counter()
        
            action_chunk = self.policy.predict_action(model_input)['action']

            inference_time = time.perf_counter() - start_time
            print(f"Inference time: {inference_time:.4f}s")
            
        return action_chunk[0], inference_time
    
    def execute_action_chunk(self, action_chunk):
        """Execute a chunk of actions with qpos jump detection"""
        start_execution = time.perf_counter()
        print("action chunk length: ", len(action_chunk))
        for i, action in enumerate(action_chunk):
            action_np = action.cpu().numpy()
            
            # Check action length and parse accordingly
            if len(action_np) == 23:
                # 23 elements: position(3) + quaternion(4) + hand_qpos(16) - DP3 format
                target_pos = action_np[:3]           # position (3)
                target_quat = action_np[3:7]         # quaternion (4) [x, y, z, w]
                leaphand_targets = action_np[7:23]   # hand qpos (16)
                
                print(f"Action {i} (23 elements): pos={target_pos}, quat={target_quat}, leaphand_range=[{np.min(leaphand_targets):.3f}, {np.max(leaphand_targets):.3f}]")
                
                # Convert to joint commands using PyBullet IK
                arm_joint_positions = self.ee_to_joint_commands(target_pos, target_quat)
                
                # Execute leaphand command using LeapNode
                try:
                    self.leap_node.set_allegro(leaphand_targets)
                    print("leap hand qpos: ", leaphand_targets)
                except Exception as e:
                    rospy.logwarn(f"Failed to execute LeapHand command for action {i}: {e}")

                # Execute arm commands
                if arm_joint_positions is not None:
                    success = self.execute_arm_joint_command(arm_joint_positions)
                    if not success:
                        rospy.logwarn(f"Failed to execute arm command for action {i}")
                else:
                    rospy.logwarn(f"No valid arm joint positions for action {i}")
                    
            elif len(action_np) == 22:
                # 22 elements: [arm_joints(6), leaphand(16)] - legacy format
                arm_joint_positions = action_np[:6]  # First 6 elements are joint angles
                leaphand_targets = action_np[6:22]   # Last 16 elements for leaphand
                
                print(f"Action {i} (22 elements): arm_joints={arm_joint_positions}, leaphand_range=[{np.min(leaphand_targets):.3f}, {np.max(leaphand_targets):.3f}]")
                
                # Execute arm commands directly with joint angles
                success = self.execute_arm_joint_command(arm_joint_positions)
                if not success:
                    rospy.logwarn(f"Failed to execute arm command for action {i}")
                
                # Execute leaphand command using LeapNode
                try:
                    self.leap_node.set_allegro(leaphand_targets)
                    print("leap hand qpos: ", leaphand_targets)
                except Exception as e:
                    rospy.logwarn(f"Failed to execute LeapHand command for action {i}: {e}")
                    
            elif len(action_np) == 25:
                # 25 elements: [ee_pos(3), ee_rot_6d(6), leaphand(16)] - legacy format
                target_pos = action_np[:3]
                target_rot_6d = action_np[3:9]
                leaphand_targets = action_np[9:25]  # 16 DOF for leaphand
                
                # Convert 6D rotation to quaternion
                if self.six_to_quat is not None:
                    target_quat = self.six_to_quat.forward(torch.tensor(target_rot_6d)[None])[0].cpu().numpy()
                else:
                    rospy.logerr("Cannot process 6D rotation format - rotation transformer not available")
                    continue
                
                print(f"Action {i} (25 elements): pos={target_pos}, quat={target_quat[:4]}, leaphand_range=[{np.min(leaphand_targets):.3f}, {np.max(leaphand_targets):.3f}]")
                
                # Convert to joint commands using PyBullet IK
                arm_joint_positions = self.ee_to_joint_commands(target_pos, target_quat)
                
                # Execute leaphand command using LeapNode
                try:
                    self.leap_node.set_allegro(leaphand_targets)
                    print("leap hand qpos: ", leaphand_targets)
                except Exception as e:
                    rospy.logwarn(f"Failed to execute LeapHand command for action {i}: {e}")

                # Execute arm commands
                if arm_joint_positions is not None:
                    success = self.execute_arm_joint_command(arm_joint_positions)
                    if not success:
                        rospy.logwarn(f"Failed to execute arm command for action {i}")
                else:
                    rospy.logwarn(f"No valid arm joint positions for action {i}")
            else:
                rospy.logerr(f"Unsupported action length: {len(action_np)} elements. Expected 23 (DP3), 22, or 25.")
                continue
            
            # Wait for execution
            rospy.sleep(config.ACTION_EXECUTION_DELAY)
            
        execution_time = time.perf_counter() - start_execution
        print(f"Execution time: {execution_time:.4f}s")
        
        return execution_time
    
    def run(self):
        """Main execution loop"""
        rospy.loginfo("Starting diffusion policy execution...")
        
        print(f"🚀 DP3 Policy 开始执行 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        rate = rospy.Rate(config.CONTROL_FREQUENCY)
        
        try:
            while not rospy.is_shutdown():
                try:
                    # 记录当前迭代开始时间
                    iteration_start_time = time.perf_counter()
                    observation_start_time = time.perf_counter()
                    
                    # Update PyBullet joint states from real robot
                    current_joints = self.kinova_node.get_current_joint_angles()
                    if current_joints is not None:
                        self.current_joint_states = current_joints
                        # Update PyBullet simulation with real robot arm qpos
                        if self.pybullet_sync_enabled:
                            self.update_pybullet_arm_qpos(current_joints)
                    
                    # Update LeapHand states from direct reading
                    try:
                        current_leaphand_states = self.leap_node.read_pos()
                        if current_leaphand_states is not None:
                            self.current_leaphand_states = current_leaphand_states
                    except Exception as e:
                        rospy.logwarn_throttle(5, f"Failed to read LeapHand position: {e}")
                        self.current_leaphand_states = None
                    
                    # Check if we have all required data
                    if (self.current_joint_states is None or 
                        self.current_leaphand_states is None or 
                        self.k4a is None):
                        rospy.logwarn_throttle(5, "Waiting for sensor data...")
                        rate.sleep()
                        continue

                    observation_time = time.perf_counter() - observation_start_time

                    # Predict action chunk
                    action_chunk, inference_time = self.predict_action()

                    if action_chunk is not None:
                        # Execute action chunk
                        execution_time = self.execute_action_chunk(action_chunk)
                        
                        # 计算总迭代时间
                        total_iteration_time = time.perf_counter() - iteration_start_time
                        
                        # 更新统计数据
                        self.step_count += 1
                        
                        # 打印每次迭代的时间统计
                        total_elapsed = time.time() - self.timing_stats['start_time']
                        print(f"Iteration #{self.step_count:4d} | "
                              f"Total: {total_iteration_time:.4f}s | "
                              f"Inference: {inference_time:.4f}s | "
                              f"Execution: {execution_time:.4f}s | "
                              f"Obs: {observation_time:.4f}s | "
                              f"Elapsed: {total_elapsed:.1f}s")
                        
                    else:
                        rospy.logwarn_throttle(5, "No action predicted, waiting for valid input...")
                    
                    rate.sleep()
                    
                except Exception as e:
                    rospy.logerr(f"Error in main loop: {e}")
                    continue
                    
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt received")
        finally:
            # 打印最终时间统计
            total_time = time.time() - self.timing_stats['start_time']
            print(f"\n🛑 DP3 Policy 执行结束 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📊 总执行时间: {total_time:.2f}s | 总迭代次数: {self.step_count}")
            if self.step_count > 0:
                avg_frequency = self.step_count / total_time
                print(f"📈 平均频率: {avg_frequency:.2f} Hz")
            self.cleanup()
        
        rospy.loginfo("Diffusion policy execution stopped")


def main():
    """Main function to run DP3 policy deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy DP3 policy on real robot system')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to DP3 checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run the model on (default: cuda:0)')
    
    args = parser.parse_args()
    
    # Check if checkpoint file exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist")
        return
    
    try:
        # Create and run the DP3 policy deployment system
        print(f"🤖 Initializing DP3 Policy Deployment System")
        print(f"📂 Checkpoint: {args.checkpoint}")
        print(f"🖥️ Device: {args.device}")
        
        system = RealSystemDP3Policy(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        
        print(f"✅ System initialized successfully")
        print(f"🎯 Starting policy execution...")
        
        # Run the system
        system.run()
        
    except Exception as e:
        print(f"❌ Error running DP3 policy deployment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 DP3 Policy Deployment System stopped")


if __name__ == "__main__":
    main()