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
from collections import defaultdict
import pybullet as p
import actionlib
import kinova_msgs.msg
from kinova_msgs.msg import JointTorque
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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
        Initialize the real system diffusion policy with PyBullet IK and ROS Image
        
        Args:
            checkpoint_path: Path to the trained diffusion policy checkpoint
            device: Device to run the model on
        """
        self.device = torch.device(device)
        
        # Initialize ROS node first
        rospy.init_node('diffusion_policy_real_system', anonymous=True)
        
        # Initialize ROS Image handling for Insta360 Air
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/insta360/image_raw", Image, self.image_callback)
        self.current_camera_image = None
        
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

        # State variables
        self.current_joint_states = None
        self.current_leaphand_states = None
        self.current_depth_image = None
        self.past_obs = None
        self.current_obs = None
        self.second_to_last_obs = None

        # Jump detection variables - similar to rollout_policy.py
        self.previous_arm_qpos = None
        self.previous_leaphand_qpos = None
        self.arm_jump_threshold = 0.5  # radians, same as rollout_policy.py
        
        # Rotation transformers
        self.mat_to_quat = RotationTransformer('matrix', 'quaternion')
        self.six_to_quat = RotationTransformer('rotation_6d', 'quaternion')
        self.quat_to_mat = RotationTransformer('quaternion', 'matrix')
        self.mode = mode
        # PyBullet qpos synchronization status
        self.pybullet_sync_enabled = True  # Set to False to disable PyBullet qpos updates

        self.previous_mat = None  
        self.rollout_step = 0      
        self.base_mat = None
        self.calculate_base_pose()
        
        self.step_num = 0
        print("Diffusion Policy Real System initialized")
    
    def calculate_base_pose(self):
        ee_pos, ee_rot = self.get_ee_pose_from_pybullet()
        self.base_mat = np.eye(4)
        self.base_mat[:3, :3] = ee_rot
        self.base_mat[:3, 3] = ee_pos
        self.previous_mat = self.base_mat.copy()
        print("Base pose matrix: ", self.base_mat)
        return
    
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
    
    def image_callback(self, msg):
        """Callback for ROS Image messages from Insta360 Air camera"""
        try:
            # print("receive image")
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            
            # Thread-safe update of current image
            self.current_camera_image = cv_image.copy()
            # print("image: ", self.current_camera_image)
                
        except Exception as e:
            rospy.logerr(f"Failed to convert ROS Image to OpenCV format: {e}")
    
    def get_camera_image(self):
        """Get the latest image from the camera"""
        if self.current_camera_image is None:
            return None
            
        try:
            # Return color image (depth not available for Insta360)
            return self.current_camera_image.copy()
            
        except Exception as e:
            rospy.logerr(f"Error getting camera image: {e}")
            return None
    
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
    
    def get_ee_pose_from_pybullet(self):
        """Get end-effector pose from PyBullet simulation"""
        try:
            link_state = p.getLinkState(self.kinova_id, config.KINOVA_END_EFFECTOR_INDEX)
            ee_pos = np.array(link_state[4])  # Position
            ee_quat = np.array(link_state[5])  # Quaternion (x, y, z, w)
            ee_quat = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])  # Rearrange to (w, x, y, z)
            # Convert quaternion to rotation matrix using RotationTransformer
            ee_quat_torch = torch.tensor(ee_quat, dtype=torch.float32, device=self.device)[None]
            ee_rot = self.quat_to_mat.forward(ee_quat_torch)[0].cpu().numpy()
            
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
        # normalized = torch.tensor(resized).float()
        
        return normalized
    
    def prepare_model_input(self):
        """Prepare input for the diffusion policy model"""
        # Get camera images
        # time.sleep(1)
        color_image = self.get_camera_image()
        
        if (color_image is None or 
            self.current_joint_states is None or 
            self.current_leaphand_states is None):
            return None
        
        # Get current end-effector pose from PyBullet
        ee_pos, ee_rot = self.get_ee_pose_from_pybullet()
        if ee_pos is None:
            return None
        
        if self.mode == 'pose':
            # If we have past observation, stack them
            if self.past_obs is not None and self.current_obs is not None:              
                model_input = {}
                model_input['agentview_image'] = torch.stack([
                    self.past_obs['agentview_image'], 
                    self.current_obs['agentview_image']
                ])
                model_input['eef_pos'] = torch.stack([
                    self.past_obs['eef_pos'], 
                    self.current_obs['eef_pos']
                ])
                model_input['eef_quat'] = torch.stack([
                    self.past_obs['eef_quat'], 
                    self.current_obs['eef_quat']
                ])
                model_input['gripper_qpos'] = torch.stack([
                    self.past_obs['gripper_qpos'], 
                    self.current_obs['gripper_qpos']
                ])
                print("model input:")
                print("past eef pos: ", self.past_obs['eef_pos'])
                print("current eef pos: ", self.current_obs['eef_pos'])
            else:
                # Prepare current observation
                init_eef_pos = np.array([0.0, 0.0, 0.0])
                init_eef_quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
                current_obs = {
                    'agentview_image': self.preprocess_image(color_image),
                    'eef_pos': torch.tensor(init_eef_pos, dtype=torch.float32),
                    'eef_quat': torch.tensor(init_eef_quat, dtype=torch.float32),
                    'gripper_qpos': torch.tensor(self.current_leaphand_states, dtype=torch.float32)
                }
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
    
    def convert_relative_to_absolute(self, relative_pos, relative_quat):
        """Convert relative end-effector pose to absolute pose in world frame"""
        relative_rot_mat = self.quat_to_mat.forward(torch.tensor(relative_quat, dtype=torch.float32, device=self.device)[None])[0].cpu().numpy()
        relative_pose_mat = np.eye(4)
        relative_pose_mat[:3, :3] = relative_rot_mat
        relative_pose_mat[:3, 3] = relative_pos
        
        absolute_pose_mat = self.previous_mat @ relative_pose_mat
        target_pos = absolute_pose_mat[:3, 3]
        target_rot_mat = absolute_pose_mat[:3, :3]
        target_quat = self.mat_to_quat.forward(torch.tensor(target_rot_mat[None], dtype=torch.float32, device=self.device))[0].cpu().numpy()
        return target_pos, target_quat # wxyz
    
    def convert_absolute_to_relative(self, absolute_pos, absolute_rot):
        """Convert absolute end-effector pose to relative pose in base frame"""
        absolute_pose_mat = np.eye(4)
        absolute_pose_mat[:3, :3] = absolute_rot
        absolute_pose_mat[:3, 3] = absolute_pos
        
        relative_pose_mat = np.linalg.inv(self.previous_mat) @ absolute_pose_mat
        relative_pos = relative_pose_mat[:3, 3]
        relative_rot_mat = relative_pose_mat[:3, :3]
        relative_quat = self.mat_to_quat.forward(torch.tensor(relative_rot_mat[None], dtype=torch.float32, device=self.device))[0].cpu().numpy()
        return relative_pos, relative_quat
    
    def execute_action_chunk(self, action_chunk):
        """Execute a chunk of actions with qpos jump detection"""
        start_execution = time.perf_counter()
        num_actions = len(action_chunk)
        print("current_base_pos: ", self.previous_mat[:, 3])
        # exit()
        for i, action in enumerate(action_chunk):
            action_np = action.cpu().numpy()
            
            # Check action length and parse accordingly
            if len(action_np) == 22:
                # 22 elements: [arm_joints(6), leaphand(16)]
                arm_joint_positions = action_np[:6]  # First 6 elements are joint angles
                leaphand_targets = action_np[6:22]   # Last 16 elements for leaphand
                
                print(f"Action {i} (22 elements): arm_joints={arm_joint_positions}, leaphand_range=[{np.min(leaphand_targets):.3f}, {np.max(leaphand_targets):.3f}]")
                
                # Execute arm commands directly with joint angles
                success = self.execute_arm_joint_command(arm_joint_positions)
                if not success:
                    rospy.logwarn(f"Failed to execute arm command for action {i}")
                
                # Execute leaphand command using LeapNode
                try:
                    self.leap_node.set_leap(leaphand_targets)
                    # self.leap_node.set_allegro(leaphand_targets)
                    print("leap hand qpos: ", leaphand_targets)
                except Exception as e:
                    rospy.logwarn(f"Failed to execute LeapHand command for action {i}: {e}")
                    
            elif len(action_np) == 25:
                # 25 elements: [ee_pos(3), ee_rot_6d(6), leaphand(16)] - original code
                relative_pos = action_np[:3]
                relative_rot_6d = action_np[3:9]
                leaphand_targets = action_np[9:25]  # 16 DOF for leaphand
                
                # Convert 6D rotation to quaternion
                relative_quat = self.six_to_quat.forward(torch.tensor(relative_rot_6d)[None])[0].cpu().numpy() # wxyz

                print(f"Action {i} (25 elements): relative pos={relative_pos}, quat={relative_quat}")

                target_pos, target_quat = self.convert_relative_to_absolute(relative_pos, relative_quat) # wxyz
                target_quat = np.array([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])  # Convert to (x,y,z,w) for PyBullet
                
                print(f"Action {i} (25 elements): target pos={target_pos}, quat={target_quat}")
                # continue
                # Convert to joint commands using PyBullet IK
                arm_joint_positions = self.ee_to_joint_commands(target_pos, target_quat)
                
                # Execute leaphand command using LeapNode
                try:
                    self.leap_node.set_leap(leaphand_targets)
                    # self.leap_node.set_allegro(leaphand_targets)
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

            self.rollout_step += 1
            # update self.previous_mat
            if self.rollout_step % 8 == 0:
                ee_pos, ee_rot = self.get_ee_pose_from_pybullet()
                self.previous_mat = np.eye(4)
                self.previous_mat[:3, :3] = ee_rot
                self.previous_mat[:3, 3] = ee_pos
                print("index in action chunk: ", i)

            if i == num_actions - 1:
                break
        # exit()
        # if self.step_num == 2:
            # exit()
        self.step_num += 1

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
            color_image = self.get_camera_image()
            # print("color image: ", color_image)
            
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
            
            if self.mode == 'pose':
                relative_pos, relative_quat = self.convert_absolute_to_relative(ee_pos, ee_rot)
                # 准备观测
                obs = {
                    'agentview_image': self.preprocess_image(color_image),
                    'eef_pos': torch.tensor(relative_pos, dtype=torch.float32),
                    'eef_quat': torch.tensor(relative_quat, dtype=torch.float32),
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
                self.past_obs = obs
            else:
                self.current_obs = obs
                
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
                        self.current_leaphand_states is None or 
                        self.current_camera_image is None):
                        rospy.logwarn_throttle(5, "Waiting for sensor data...s")
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