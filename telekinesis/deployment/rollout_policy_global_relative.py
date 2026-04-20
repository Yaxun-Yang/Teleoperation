#!/usr/bin/env python3
"""
Real-time deployment of diffusion policy on real robot system
Robot: Kinova arm (6DOF) + LeapHand (16DOF)
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
from cv_bridge import CvBridge
from collections import defaultdict
from scipy.spatial.transform import Rotation
import pybullet as p
import actionlib
import kinova_msgs.msg
from kinova_msgs.msg import JointTorque
from sensor_msgs.msg import Image
import threading
# import pyk4a
# from pyk4a import Config, PyK4A

# Add necessary paths
import path_config
sys.path.append(path_config.DIFFUSION_POLICY_DIR)
sys.path.append(path_config.SRC_DIR)

from diffusion_policy.model.common.rotation_transformer import RotationTransformer
import real_system_config as config
from kinova_node import KinovaNode

# Add LeapHand utilities
sys.path.append(os.path.join(path_config.SRC_DIR, 'leap_hand_utils'))
from leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils.leap_hand_utils as lhu


class LeapNode:
    """LeapHand control node - copied from leap_kinova_ik_real_recording.py"""
    def __init__(self):
        # Some parameters
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(
            np.zeros(16))

        # You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        try:
            self.dxl_client = DynamixelClient(motors, "/dev/ttyUSB0", 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(
                    motors, "/dev/ttyUSB1", 4000000)
                self.dxl_client.connect()
            except Exception:
                try:
                    self.dxl_client = DynamixelClient(
                        motors, "/dev/ttyUSB2", 4000000)
                    self.dxl_client.connect()
                except Exception:
                    self.dxl_client = DynamixelClient(motors, "COM13", 4000000)
                    self.dxl_client.connect()
        # Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kP, 84, 2
        )  # Pgain stiffness
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2
        )  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kI, 82, 2
        )  # Igain
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kD, 80, 2
        )  # Dgain damping
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2
        )  # Dgain damping for side to side should be a bit less
        # Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(
            len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        print("Leap Node initialized")

    # Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # allegro compatibility
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # read position
    def read_pos(self):
        return self.dxl_client.read_pos()  # 16dof

    # read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()

    # read current
    def read_cur(self):
        return self.dxl_client.read_cur()


class RealSystemDiffusionPolicy:
    def __init__(self, checkpoint_path, device='cuda:0', mode='pose'):
        """
        Initialize the real system diffusion policy with PyBullet IK and Azure Kinect
        
        Args:
            checkpoint_path: Path to the trained diffusion policy checkpoint
            device: Device to run the model on
        """
        self.device = torch.device(device)
        
        # Initialize ROS node first
        rospy.init_node('diffusion_policy_real_system', anonymous=True)
        
        # # Initialize Azure Kinect camera
        # self.init_azure_kinect()
        
        # Load diffusion policy model
        self.load_policy(checkpoint_path)
        
        # Initialize ROS interfaces
        self.init_ros_interfaces()
        
        # Initialize Kinova node
        self.kinova_node = KinovaNode()
        
        # Initialize LeapHand node
        self.leap_node = LeapNode()
        
        # Initialize PyBullet for IK
        self.init_pybullet()

        # Initialize Insta360 camera
        self.init_insta360()

        # State variables
        self.current_joint_states = None
        self.current_leaphand_states = None
        self.current_camera_image = None
        self.current_depth_image = None
        self.current_insta_image = None
        self.past_obs = None
        self.last_pos = None
        self.first_trial = True
        
        # 新增：用于存储action_chunk执行过程中倒数第二个动作后的观测
        self.second_to_last_obs = None
        
        # Jump detection variables - similar to rollout_policy.py
        self.previous_arm_qpos = None
        self.previous_leaphand_qpos = None
        self.arm_jump_threshold = 0.5  # radians, same as rollout_policy.py
        
        # Rotation transformers
        self.mat_to_quat = RotationTransformer('matrix', 'quaternion')
        self.six_to_quat = RotationTransformer('rotation_6d', 'quaternion')
        self.mode = mode
        # PyBullet qpos synchronization status
        self.pybullet_sync_enabled = True  # Set to False to disable PyBullet qpos updates
        
        self.base_pose_mat = None  # 用于存储初始位姿的4x4矩阵
        self.calculate_base_pose()

        self.insta_lock = threading.Lock()
        
        print("Diffusion Policy Real System initialized")
    
    def calculate_base_pose(self):
        """计算初始位姿并存储为4x4矩阵"""
        try:
            # 获取当前末端执行器的位姿
            ee_pos, ee_rot = self.get_ee_pose_from_pybullet()
            if ee_pos is None or ee_rot is None:
                rospy.logwarn("Failed to get initial EE pose, using identity matrix")
                self.base_pose_mat = np.eye(4)
                return

            # 构造4x4初始位姿矩阵
            self.base_pose_mat = np.eye(4)
            self.base_pose_mat[:3, :3] = ee_rot  # 旋转部分
            self.base_pose_mat[:3, 3] = ee_pos  # 平移部分
            print("Base pose matrix calculated:", self.base_pose_mat)
        except Exception as e:
            rospy.logerr(f"Error calculating base pose: {e}")
            self.base_pose_mat = np.eye(4)  # 使用单位矩阵作为默认值

    def relative_to_absolute_pose(self, relative_pos, relative_rot_6d):
        """
        将相对位姿转换为绝对位姿

        Args:
            relative_pos: 相对位置 (3,)
            relative_rot_6d: 相对旋转的6D表示 (6,)

        Returns:
            absolute_pose_mat: 绝对位姿的4x4矩阵
        """
        try:
            # 将6D旋转转换为3x3旋转矩阵
            relative_rot_mat = self.six_to_quat.forward(torch.tensor(relative_rot_6d)[None])
            relative_rot_mat = Rotation.from_quat(relative_rot_mat[0].cpu().numpy()).as_matrix()

            # 构造相对位姿的4x4矩阵
            relative_pose_mat = np.eye(4)
            relative_pose_mat[:3, :3] = relative_rot_mat
            relative_pose_mat[:3, 3] = relative_pos

            # 计算绝对位姿
            absolute_pose_mat = self.base_pose_mat @ relative_pose_mat
            return absolute_pose_mat
        except Exception as e:
            rospy.logerr(f"Error converting relative to absolute pose: {e}")
            return None

    def init_pybullet(self):
        """Initialize PyBullet simulation for IK"""
        # Connect to PyBullet (use DIRECT mode for headless operation, GUI mode for visualization)
        # Set to p.GUI to visualize the real-time qpos updates
        p.connect(p.GUI)  # Change to p.GUI to see the robot moving in PyBullet
        
        # Load Kinova robot
        path_src = os.path.dirname(os.path.abspath(__file__))
        
        self.kinova_id = p.loadURDF(
            config.KINOVA_URDF_PATH,
            [0.0, 0.0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        
        self.num_joints = p.getNumJoints(self.kinova_id)
        
        # Initialize joints according to real robot's current position
        try:
            # Get current joint positions from real robot
            current_joint_angles = self.kinova_node.get_current_joint_angles()
            
            if current_joint_angles is not None and len(current_joint_angles) == 6:
                print("Initializing PyBullet with real robot joint positions:")
                for i, joint_angle in enumerate(current_joint_angles):
                    p.resetJointState(self.kinova_id, i + 2, joint_angle)  # joints 2-7 for 6DOF arm
                    print(f"  Joint {i+1}: {joint_angle:.4f} rad")
            else:
                print("Could not get real robot joint positions, using default configuration")
                # Initialize joints to a reasonable default configuration
                for i in range(2, 8):
                    p.resetJointState(self.kinova_id, i, 0.0)
        except Exception as e:
            print(f"Error getting real robot joint positions: {e}")
            print("Using default joint configuration")
            # Initialize joints to a reasonable default configuration
            for i in range(2, 8):
                p.resetJointState(self.kinova_id, i, 0.0)
        
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        
        print("PyBullet initialized for IK")
    
    def init_insta360(self):
        # Subscribe to the Insta360 Air camera topic
        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/insta360/image_raw',
            Image,
            self.insta360_callback,
            queue_size=1
        )

    def insta360_callback(self, msg):
        """Callback function for processing incoming camera images"""
        try:
            # Convert ROS image message to OpenCV image with RGB format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            # print("fisheye img shape: ", cv_image.shape)
            
            # Thread-safe update of current image
            with self.insta_lock:
                self.current_insta_image = cv_image.copy()
                
        except Exception as e:
            print(f"Camera callback error: {e}")

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
            
            # Get camera calibration
            calibration = self.k4a.calibration
            self.K = calibration.get_camera_matrix(1)  # Color camera matrix
            
            # Calculate downscale factor
            capture = self.k4a.get_capture()
            H, W = capture.color.shape[:2]
            self.downscale = config.SHORTER_SIDE / min(H, W)
            self.target_H = int(H * self.downscale)
            self.target_W = int(W * self.downscale)
            self.K[:2] *= self.downscale
            
            print(f"Azure Kinect initialized: {W}x{H} -> {self.target_W}x{self.target_H}")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize Azure Kinect: {e}")
            self.k4a = None
    
    def load_policy(self, checkpoint_path):
        """Load the trained diffusion policy model"""
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        # print(cfg)
        # exit()
        workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir='data')
        workspace.load_payload(payload)
        
        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model
        
        self.policy.to(self.device)
        self.policy.eval()
        print(f"Policy loaded from {checkpoint_path}")
    
    def init_ros_interfaces(self):
        """Initialize ROS subscribers and publishers"""
        # No need for LeapHand ROS interfaces since we use direct serial communication
        # Wait for initial data
        rospy.sleep(2.0)
        print("ROS interfaces initialized")
    
    def joint_callback(self, msg):
        """Callback for joint states - now handled by KinovaNode"""
        pass  # This is now handled by the KinovaNode
    
    def get_insta_images(self):
        with self.insta_lock:
            if self.current_insta_image is not None:
                return self.current_insta_image.copy()
            else:
                return None

    def get_camera_image(self):
        """Get image from Azure Kinect camera"""
        if self.k4a is None:
            return None, None
        
        try:
            # time.sleep(1)
            capture = self.k4a.get_capture(timeout=33)
            
            # Get color image
            color = capture.color[..., :3].astype(np.uint8)
            color = cv2.resize(color, (self.target_W, self.target_H), interpolation=cv2.INTER_NEAREST)
            
            # Get depth image
            depth = capture.transformed_depth.astype(np.float32) / 1e3
            depth = cv2.resize(depth, (self.target_W, self.target_H), interpolation=cv2.INTER_NEAREST)
            depth[(depth < 0.01) | (depth >= config.ZFAR)] = 0
            
            return color, depth
            
        except Exception as e:
            rospy.logerr(f"Error getting camera image: {e}")
            return None, None
    
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
    
    def detect_qpos_jump(self, target_qpos, current_qpos, joint_type="arm"):
        """
        Detect jumps in joint positions (qpos) - similar to rollout_policy.py
        
        Args:
            target_qpos: Target joint positions from IK
            current_qpos: Current actual joint positions  
            joint_type: "arm" or "leaphand"
        
        Returns:
            bool: True if jump detected, False otherwise
        """
        if target_qpos is None or current_qpos is None:
            return False
        
        if len(target_qpos) != len(current_qpos):
            rospy.logwarn(f"{joint_type} qpos size mismatch: target={len(target_qpos)}, current={len(current_qpos)}")
            return False
        
        # Calculate joint position error - same as rollout_policy.py
        if joint_type == "arm":
            qpos_err = np.linalg.norm(target_qpos - current_qpos)
            
            if qpos_err > self.arm_jump_threshold:
                print("Jump detected. Joint error {}. This is likely caused when hardware detects something unsafe.".format(qpos_err))
                return True
        
        return False
    
    def preprocess_image(self, image):
        """Preprocess camera image for the model"""
        if image is None:
            return None
        
        # Crop margins if specified
        cropped_image = image
        if config.CROP_MARGINS[0] > 0:
            # cropped_image = image[:, config.CROP_MARGINS[0]:-config.CROP_MARGINS[1]]
            cropped_image = image[:, :]
        
        # Resize image
        resized = cv2.resize(cropped_image, config.IMAGE_SIZE)
        
        # # Create a larger display version for better visibility
        # display_scale = 5  # Scale factor to enlarge the window
        # display_height, display_width = resized.shape[:2]
        # enlarged_display = cv2.resize(resized, 
        #                              (display_width * display_scale, display_height * display_scale), 
        #                              interpolation=cv2.INTER_LINEAR)
        
        # # Create named window with resizable property
        # cv2.namedWindow('Resized Image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Resized Image', display_width * display_scale, display_height * display_scale)
        
        # # Visualize the enlarged image
        # cv2.imshow('Resized Image', enlarged_display)
        # cv2.waitKey(1)

        # Normalize to [0,1] range
        normalized = torch.tensor(resized).permute(2, 0, 1).float() / 255.0

        return normalized
    
    def prepare_model_input(self):
        """Prepare input for the diffusion policy model"""
        # Get camera images
        # time.sleep(1)
        color_image = self.get_insta_images()
        time.sleep(0.5)
        if (color_image is None or 
            self.current_joint_states is None or 
            self.current_leaphand_states is None):
            return None
        
        # Get current end-effector pose from PyBullet
        ee_pos, ee_rot = self.get_ee_pose_from_pybullet()
        if ee_pos is None:
            return None
        
        if self.mode == 'pose':
            # 计算相对位姿
            ee_pose_mat = np.eye(4)
            ee_pose_mat[:3, :3] = ee_rot
            ee_pose_mat[:3, 3] = ee_pos
            
            # 计算相对变换矩阵
            relative_pose_mat = np.linalg.inv(self.base_pose_mat) @ ee_pose_mat
            
            # 提取相对位置和旋转
            relative_pos = relative_pose_mat[:3, 3]
            relative_rot = relative_pose_mat[:3, :3]
            
            if self.first_trial:
                relative_pos = np.zeros(3)
                relative_rot = np.eye(3)
                self.first_trial = False
            print("current relative pose: ", relative_pos, relative_rot)
            # Prepare current observation with relative poses
            current_obs = {
                'agentview_image': self.preprocess_image(color_image),
                'eef_pos': torch.tensor(relative_pos, dtype=torch.float32),
                'eef_quat': self.mat_to_quat.forward(torch.tensor(relative_rot[None])).squeeze(0),
                'gripper_qpos': torch.tensor(self.current_leaphand_states, dtype=torch.float32)
            }

            # If we have past observation, stack them
            if self.past_obs is not None:
                model_input = {}
                model_input['agentview_image'] = torch.stack([
                    self.past_obs['agentview_image'], 
                    current_obs['agentview_image']
                ])
                model_input['eef_pos'] = torch.stack([
                    self.past_obs['eef_pos'], 
                    current_obs['eef_pos']
                ])
                model_input['eef_quat'] = torch.stack([
                    self.past_obs['eef_quat'], 
                    current_obs['eef_quat']
                ])
                model_input['gripper_qpos'] = torch.stack([
                    self.past_obs['gripper_qpos'], 
                    current_obs['gripper_qpos']
                ])
            else:
                # First step, duplicate current observation
                model_input = {}
                model_input['agentview_image'] = torch.stack([
                    current_obs['agentview_image'], 
                    current_obs['agentview_image']
                ])
                model_input['eef_pos'] = torch.stack([
                    current_obs['eef_pos'], 
                    current_obs['eef_pos']
                ])
                model_input['eef_quat'] = torch.stack([
                    current_obs['eef_quat'], 
                    current_obs['eef_quat']
                ])
                model_input['gripper_qpos'] = torch.stack([
                    current_obs['gripper_qpos'], 
                    current_obs['gripper_qpos']
                ])
        else:
            # Prepare current observation
            current_obs = {
                'agentview_image': self.preprocess_image(color_image),
                'arm_qpos': torch.tensor(self.current_joint_states, dtype=torch.float32),
                'gripper_qpos': torch.tensor(self.current_leaphand_states, dtype=torch.float32)
            }

            # If we have past observation, stack them
            if self.past_obs is not None:
                model_input = {}
                model_input['agentview_image'] = torch.stack([
                    self.past_obs['agentview_image'], 
                    current_obs['agentview_image']
                ])
                model_input['arm_qpos'] = torch.stack([
                    self.past_obs['arm_qpos'], 
                    current_obs['arm_qpos']
                ])
                model_input['gripper_qpos'] = torch.stack([
                    self.past_obs['gripper_qpos'], 
                    current_obs['gripper_qpos']
                ])
            else:
                # First step, duplicate current observation
                model_input = {}
                model_input['agentview_image'] = torch.stack([
                    current_obs['agentview_image'], 
                    current_obs['agentview_image']
                ])
                model_input['arm_qpos'] = torch.stack([
                    current_obs['arm_qpos'], 
                    current_obs['arm_qpos']
                ])
                model_input['gripper_qpos'] = torch.stack([
                    current_obs['gripper_qpos'], 
                    current_obs['gripper_qpos']
                ])

        # Update past observation
        self.past_obs = current_obs

        # Add batch dimension
        model_input = {key: value[None].to(self.device) for key, value in model_input.items()}

        return model_input
    
    def predict_action(self):
        """Predict action using diffusion policy"""
        model_input = self.prepare_model_input()
        if model_input is None:
            return None

        with torch.no_grad():
            start_time = time.perf_counter()
        
            action_chunk = self.policy.predict_action(model_input)['action']

            inference_time = time.perf_counter() - start_time
            print(f"Inference time: {inference_time:.4f}s")
        return action_chunk[0]  # Remove batch dimension
    
    def execute_action_chunk(self, action_chunk):
        """Execute a chunk of actions and record states for the last two actions"""
        start_execution = time.perf_counter()
        
        # num_actions = len(action_chunk)
        num_actions = 2
   
        for i, action in enumerate(action_chunk):
            action_np = action.cpu().numpy()
            
            # Check action length and parse accordingly
            if len(action_np) == 22:
                # 22 elements: [arm_joints(6), leaphand(16)]
                arm_joint_positions = action_np[:6]
                leaphand_targets = action_np[6:22]
                
                print(f"Action {i} (22 elements): arm_joints={arm_joint_positions}, leaphand_range=[{np.min(leaphand_targets):.3f}, {np.max(leaphand_targets):.3f}]")
                
                # Execute arm commands directly with joint angles
                success = self.execute_arm_joint_command(arm_joint_positions)
                if not success:
                    rospy.logwarn(f"Failed to execute arm command for action {i}")
                
                # Execute leaphand command using LeapNode
                try:
                    self.leap_node.set_leap(leaphand_targets)
                    print("leap hand qpos: ", leaphand_targets)
                except Exception as e:
                    rospy.logwarn(f"Failed to execute LeapHand command for action {i}: {e}")
                    
            elif len(action_np) == 25:
                # 25 elements: [ee_pos(3), ee_rot_6d(6), leaphand(16)]
                relative_pos = action_np[:3]
                relative_rot_6d = action_np[3:9]
                leaphand_targets = action_np[9:25]

                # 转换为绝对位姿
                absolute_pose_mat = self.relative_to_absolute_pose(relative_pos, relative_rot_6d)
                if absolute_pose_mat is None:
                    rospy.logwarn(f"Skipping action {i} due to invalid absolute pose")
                    continue

                # 从绝对位姿提取位置和四元数
                target_pos = absolute_pose_mat[:3, 3]
                target_quat = Rotation.from_matrix(absolute_pose_mat[:3, :3]).as_quat()

                print(f"Action {i} (25 elements): relative_pos={relative_pos}, pos={target_pos}, quat={target_quat[:4]}, leaphand_range=[{np.min(leaphand_targets):.3f}, {np.max(leaphand_targets):.3f}]")
                diff_pos = relative_pos - self.last_pos if self.last_pos is not None else relative_pos
                print(f"Action {i} (25 elements): Position Difference: {diff_pos}")
                self.last_pos = relative_pos
                arm_joint_positions = self.ee_to_joint_commands(target_pos, target_quat)
                
                # Execute leaphand command using LeapNode
                try:
                    self.leap_node.set_leap(leaphand_targets)
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
                rospy.logerr(f"Unsupported action length: {len(action_np)} elements. Expected 22 or 25.")
                continue
            
            # Wait for execution
            rospy.sleep(config.ACTION_EXECUTION_DELAY)
            
            # 记录倒数第二个动作后的状态 (i == num_actions - 2)
            if i == num_actions - 2:
                self.record_observation_for_next_inference(is_second_to_last=True)
            
            # 记录倒数第一个动作后的状态 (i == num_actions - 1)
            elif i == num_actions - 1:
                self.record_observation_for_next_inference(is_second_to_last=False)
                break
        
        print(f"Execution time: {time.perf_counter()-start_execution}")
        return True
    
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
            
            # Optional: Print debug info (can be commented out for performance)
            # rospy.logdebug(f"PyBullet arm qpos updated: {joint_positions}")
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Error updating PyBullet arm qpos: {e}")
            return False

    def record_observation_for_next_inference(self, is_second_to_last=False):
        """
        记录观测状态用于下一次推理
        
        Args:
            is_second_to_last: True表示这是倒数第二个动作后的状态，False表示倒数第一个动作后的状态
        """
        try:
            # 获取相机图像
            color_image = self.get_insta_images()
            # time.sleep(0.5)
            print("color image: ", color_image)
            
            if color_image is None:
                rospy.logwarn("Failed to get camera image for observation recording")
                return
            
            # 获取当前关节状态
            current_joints = self.kinova_node.get_current_joint_angles()
            if current_joints is not None:
                self.current_joint_states = current_joints
                # 更新 PyBullet 仿真
                if self.pybullet_sync_enabled:
                    self.update_pybullet_arm_qpos(current_joints)
            
            # 获取 LeapHand 状态
            try:
                current_leaphand_states = self.leap_node.read_pos()
                if current_leaphand_states is not None:
                    self.current_leaphand_states = current_leaphand_states
            except Exception as e:
                rospy.logwarn(f"Failed to read LeapHand position: {e}")
            
            # 获取末端执行器位姿
            ee_pos, ee_rot = self.get_ee_pose_from_pybullet()
            
            if ee_pos is None:
                rospy.logwarn("Failed to get EE pose from PyBullet")
                return
            
            curr_pose_mat = np.eye(4)
            curr_pose_mat[:3, :3] = ee_rot
            curr_pose_mat[:3, 3] = ee_pos
            relative_pose_mat = np.linalg.inv(self.base_pose_mat) @ curr_pose_mat
            relative_pos = relative_pose_mat[:3, 3]
            relative_rot = relative_pose_mat[:3, :3]
            print("recording relative pose: ", relative_pos, relative_rot)
            
            if self.mode == 'pose':
                # 准备观测
                obs = {
                    'agentview_image': self.preprocess_image(color_image),
                    'eef_pos': torch.tensor(relative_pos, dtype=torch.float32),
                    'eef_quat': self.mat_to_quat.forward(torch.tensor(relative_rot[None])).squeeze(0),
                    'gripper_qpos': torch.tensor(self.current_leaphand_states, dtype=torch.float32)
                }
            else:
                # 准备观测
                obs = {
                    'agentview_image': self.preprocess_image(color_image),
                    'arm_qpos': torch.tensor(self.current_joint_states, dtype=torch.float32),
                    'gripper_qpos': torch.tensor(self.current_leaphand_states, dtype=torch.float32)
                }
            
            # 根据是倒数第二个还是倒数第一个动作来存储
            if is_second_to_last:
                self.second_to_last_obs = obs
                print("Recorded second-to-last observation")
            else:
                # 倒数第一个动作后的状态作为当前观测
                # 倒数第二个动作后的状态作为过去观测
                if self.second_to_last_obs is not None:
                    self.past_obs = self.second_to_last_obs
                    print("Updated past_obs from second_to_last_obs")
                else:
                    # 如果没有倒数第二个观测（比如action_chunk长度<2），则使用当前观测作为过去观测
                    self.past_obs = obs
                    rospy.logwarn("No second_to_last_obs available, using current obs as past_obs")
                
        except Exception as e:
            rospy.logerr(f"Error recording observation: {e}")

    def run(self):
        """Main execution loop"""
        rospy.loginfo("Starting diffusion policy execution...")
        
        rate = rospy.Rate(config.CONTROL_FREQUENCY)
        
        try:
            while not rospy.is_shutdown():
                try:
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
                            # Update PyBullet simulation with real LeapHand qpos (if loaded and enabled)
                            # if self.pybullet_sync_enabled:
                            #     self.update_pybullet_leaphand_qpos(current_leaphand_states)
                    except Exception as e:
                        rospy.logwarn_throttle(5, f"Failed to read LeapHand position: {e}")
                        self.current_leaphand_states = None
                    
                    # Check if we have all required data
                    if (self.current_joint_states is None or 
                        self.current_leaphand_states is None):
                        rospy.logwarn_throttle(5, "Waiting for sensor data...")
                        rate.sleep()
                        continue

                    # Predict action chunk
                    action_chunk = self.predict_action()

                    if action_chunk is not None:
                        # Execute action chunk
                        self.execute_action_chunk(action_chunk)
                    else:
                        rospy.logwarn_throttle(5, "No action predicted, waiting for valid input...")
                    
                    rate.sleep()
                    
                except Exception as e:
                    rospy.logerr(f"Error in main loop: {e}")
                    continue
                    
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt received")
        finally:
            self.cleanup()
        
        rospy.loginfo("Diffusion policy execution stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel any ongoing arm actions
            if hasattr(self, 'kinova_node'):
                self.kinova_node.cancel_all_goals()
            
            # # Stop Azure Kinect
            # if self.k4a is not None:
            #     self.k4a.stop()
            
            # Disconnect PyBullet
            p.disconnect()
            
            rospy.loginfo("Cleanup completed")
        except Exception as e:
            rospy.logerr(f"Error during cleanup: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy diffusion policy on real robot')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to diffusion policy checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run the system
        system = RealSystemDiffusionPolicy(args.checkpoint, args.device)
        system.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down...")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")


if __name__ == '__main__':
    main()