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
from scipy.spatial.transform import Rotation
import pybullet as p
import actionlib
import kinova_msgs.msg
from kinova_msgs.msg import JointTorque
import pyk4a
from pyk4a import Config, PyK4A
import h5py
from pathlib import Path

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
    def __init__(self, checkpoint_path, device='cuda:0', hdf5_data_path=None):
        """
        Initialize the real system diffusion policy with PyBullet IK and Azure Kinect
        
        Args:
            checkpoint_path: Path to the trained diffusion policy checkpoint
            device: Device to run the model on
            hdf5_data_path: Path to HDF5 file to read observation data from
        """
        self.device = torch.device(device)
        self.hdf5_data_path = hdf5_data_path
        self.hdf5_data = None
        self.hdf5_step_idx = 0
        
        # Load HDF5 data if provided
        if self.hdf5_data_path and Path(self.hdf5_data_path).exists():
            self.load_hdf5_data()
        
        # Initialize ROS node first
        rospy.init_node('diffusion_policy_real_system', anonymous=True)
        
        # Initialize PyBullet for IK
        self.init_pybullet()
        
        # Initialize Azure Kinect camera
        self.init_azure_kinect()
        
        # Load diffusion policy model
        self.load_policy(checkpoint_path)
        
        # Initialize ROS interfaces
        self.init_ros_interfaces()
        
        # Initialize Kinova node
        self.kinova_node = KinovaNode()
        
        # Initialize LeapHand node
        self.leap_node = LeapNode()
        
        # State variables
        self.current_joint_states = None
        self.current_leaphand_states = None
        self.current_camera_image = None
        self.current_depth_image = None
        self.past_obs = None
        
        self.ee_pos = None
        self.ee_rot = None

        # Jump detection variables - similar to rollout_policy.py
        self.previous_arm_qpos = None
        self.previous_leaphand_qpos = None
        self.arm_jump_threshold = 0.5  # radians, same as rollout_policy.py
        
        # Rotation transformers
        self.mat_to_quat = RotationTransformer('matrix', 'quaternion')
        self.six_to_quat = RotationTransformer('rotation_6d', 'quaternion')
        
        # PyBullet qpos synchronization status
        self.pybullet_sync_enabled = True  # Set to False to disable PyBullet qpos updates
        
        print("Diffusion Policy Real System initialized")
    
    def init_pybullet(self):
        """Initialize PyBullet simulation for IK"""
        # Connect to PyBullet (use DIRECT mode for headless operation, GUI mode for visualization)
        # Set to p.GUI to visualize the real-time qpos updates
        p.connect(p.DIRECT)  # Change to p.GUI to see the robot moving in PyBullet
        
        # Load Kinova robot
        path_src = os.path.dirname(os.path.abspath(__file__))
        
        self.kinova_id = p.loadURDF(
            config.KINOVA_URDF_PATH,
            [0.0, 0.0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        
        # Load LeapHand (for reference, though we might not need it for arm IK)
        # leap_urdf_path = os.path.join(path_config.TELEKINESIS_DIR, "leap_hand_mesh_right/robot_pybullet.urdf")
        # if os.path.exists(leap_urdf_path):
        #     self.leap_id = p.loadURDF(
        #         leap_urdf_path,
        #         [0.0, 0.038, 0.098],
        #         p.getQuaternionFromEuler([0, -1.57, 0]),
        #         useFixedBase=True,
        #     )
        
        self.num_joints = p.getNumJoints(self.kinova_id)
        
        # Initialize joints to a reasonable configuration
        for i in range(2, 8):
            p.resetJointState(self.kinova_id, i, 0.0)
        
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        
        print("PyBullet initialized for IK")
    
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
    
    def get_camera_image(self):
        """Get image from Azure Kinect camera or HDF5 data"""
        if self.hdf5_data is not None:
            # Read from HDF5 file
            if self.hdf5_step_idx >= self.total_steps:
                rospy.loginfo("Reached end of HDF5 data, restarting from beginning")
                self.hdf5_step_idx = 0
            
            # Get image from HDF5
            color = self.hdf5_data['agentview_image'][self.hdf5_step_idx]
            
            # For depth, we might not have it in the HDF5, so return None
            depth = None
            
            cv2.imshow('HDF5 Image Stream', color)
            cv2.waitKey(1)
            
            self.hdf5_step_idx += 1
            return color, depth
        
        else:
            # Use Azure Kinect camera (original implementation)
            if self.k4a is None:
                return None, None
            
            try:
                capture = self.k4a.get_capture()
                
                # Get color image
                color = capture.color[..., :3].astype(np.uint8)
                
                # Convert BGR to RGB (Azure Kinect gives BGR, but we want RGB)
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                
                color = cv2.resize(color, (self.target_W, self.target_H), interpolation=cv2.INTER_NEAREST)
                
                # Get depth image
                depth = capture.transformed_depth.astype(np.float32) / 1e3
                depth = cv2.resize(depth, (self.target_W, self.target_H), interpolation=cv2.INTER_NEAREST)
                depth[(depth < 0.01) | (depth >= config.ZFAR)] = 0
                
                # Show the image in real-time
                display_image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display
                cv2.imshow('Azure Kinect Stream', display_image)
                cv2.waitKey(1)
                
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
        if config.CROP_MARGINS[0] > 0:
            image = image[:, config.CROP_MARGINS[0]:-config.CROP_MARGINS[1]]
        
        # Resize and normalize
        resized = cv2.resize(image, config.IMAGE_SIZE)
        normalized = torch.tensor(resized).permute(2, 0, 1).float() / 255.0

        # Show the processed image for debugging
        display_resized = resized.copy()
        cv2.imshow('Processed Image', display_resized)
        cv2.waitKey(1)
        
        return normalized
    
    def prepare_model_input(self):
        """Prepare input for the diffusion policy model"""
        # Get camera images
        color_image, depth_image = self.get_camera_image()
        
        # Get robot states from HDF5 or hardware
        arm_qpos, gripper_qpos = self.get_current_states_from_hdf5_or_hardware()
        
        if color_image is None or arm_qpos is None or gripper_qpos is None:
            return None
        
        # Update current states
        self.current_joint_states = arm_qpos
        self.current_leaphand_states = gripper_qpos
        
        # Get current end-effector pose
        ee_pos, ee_rot = self.get_ee_pose_from_hdf5_or_pybullet()
        
        # self.ee_pos = ee_pos
        # self.ee_rot = self.mat_to_quat.forward(torch.tensor(ee_rot[None])).squeeze(0).numpy()

        if ee_pos is None:
            return None

        # Prepare current observation
        current_obs = {
            'agentview_image': self.preprocess_image(color_image),
            'eef_pos': torch.tensor(ee_pos, dtype=torch.float32),
            'eef_quat': self.mat_to_quat.forward(torch.tensor(ee_rot[None])).squeeze(0),
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
        """Execute a chunk of actions with qpos jump detection"""
        start_execution = time.perf_counter()
        
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
                else:
                    # Update PyBullet simulation with the target joint positions
                    if self.pybullet_sync_enabled:
                        self.update_pybullet_arm_qpos(arm_joint_positions)
                
                # Execute leaphand command using LeapNode
                try:
                    self.leap_node.set_allegro(leaphand_targets)
                    print("leap hand qpos: ", leaphand_targets)
                except Exception as e:
                    rospy.logwarn(f"Failed to execute LeapHand command for action {i}: {e}")
                    
            elif len(action_np) == 25:
                # 25 elements: [ee_pos(3), ee_rot_6d(6), leaphand(16)] - original code
                target_pos = action_np[:3]
                target_rot_6d = action_np[3:9]
                leaphand_targets = action_np[9:25]  # 16 DOF for leaphand
                
                # Convert 6D rotation to quaternion
                target_quat = self.six_to_quat.forward(torch.tensor(target_rot_6d)[None])[0].cpu().numpy()
                
                print(f"Action {i} (25 elements): pos={target_pos}, quat={target_quat[:4]}, leaphand_range=[{np.min(leaphand_targets):.3f}, {np.max(leaphand_targets):.3f}]")
                
                # Convert to joint commands using PyBullet IK
                arm_joint_positions = self.ee_to_joint_commands(target_pos, target_quat)
                
                # Execute arm commands
                if arm_joint_positions is not None:
                    success = self.execute_arm_joint_command(arm_joint_positions)
                    if not success:
                        rospy.logwarn(f"Failed to execute arm command for action {i}")
                else:
                    rospy.logwarn(f"No valid arm joint positions for action {i}")
                
                # Execute leaphand command using LeapNode
                try:
                    self.leap_node.set_allegro(leaphand_targets)
                    print(f"LeapHand command sent: range=[{np.min(leaphand_targets):.3f}, {np.max(leaphand_targets):.3f}]")
                except Exception as e:
                    rospy.logwarn(f"Failed to execute LeapHand command for action {i}: {e}")
            else:
                rospy.logerr(f"Unsupported action length: {len(action_np)} elements. Expected 22 or 25.")
                continue
            
            # Wait for execution
            rospy.sleep(config.ACTION_EXECUTION_DELAY)
            
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
            
            link_state = p.getLinkState(self.kinova_id, 9)
            curr_pos = link_state[4]
            curr_quat = link_state[5]
            print(f"PyBullet EE pose updated: pos={curr_pos}, quat={curr_quat}")

            # Optional: Print debug info (can be commented out for performance)
            # rospy.logdebug(f"PyBullet arm qpos updated: {joint_positions}")
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Error updating PyBullet arm qpos: {e}")
            return False
    
    # def update_pybullet_leaphand_qpos(self, leaphand_positions):
    #     """
    #     Update PyBullet simulation with real LeapHand joint positions (if LeapHand is loaded in PyBullet)
        
    #     Args:
    #         leaphand_positions: Current LeapHand joint positions (16 DOF array)
    #     """
    #     try:
    #         # Note: This is a placeholder for LeapHand PyBullet update
    #         # Currently LeapHand URDF is commented out in init_pybullet()
    #         # If you decide to load LeapHand in PyBullet, uncomment the URDF loading
    #         # and implement the joint state updates here
            
    #         if not hasattr(self, 'leap_id'):
    #             return False  # LeapHand not loaded in PyBullet
                
    #         if leaphand_positions is None or len(leaphand_positions) != 16:
    #             rospy.logwarn(f"Invalid LeapHand positions for PyBullet update: {leaphand_positions}")
    #             return False
            
    #         # Update PyBullet LeapHand joint states
    #         for i, joint_pos in enumerate(leaphand_positions):
    #             p.resetJointState(self.leap_id, i, joint_pos)
            
    #         return True
            
    #     except Exception as e:
    #         rospy.logerr(f"Error updating PyBullet LeapHand qpos: {e}")
    #         return False

    def run(self):
        """Main execution loop"""
        rospy.loginfo("Starting diffusion policy execution...")
        
        # Create OpenCV windows
        cv2.namedWindow('HDF5 Image Stream' if self.hdf5_data else 'Azure Kinect Stream', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
        
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
                    except Exception as e:
                        rospy.logwarn_throttle(5, f"Failed to read LeapHand position: {e}")
                        self.current_leaphand_states = None
                    
                    # Check for 'q' key to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        rospy.loginfo("'q' key pressed, shutting down...")
                        break
                    
                    # If using HDF5 data and reached the end, you might want to loop or stop
                    if self.hdf5_data and self.hdf5_step_idx >= self.total_steps:
                        rospy.loginfo("Finished processing all HDF5 data")
                        break
                    
                    # Check if we have all required data
                    if ((self.hdf5_data is None and (self.current_joint_states is None or 
                        self.current_leaphand_states is None or self.k4a is None)) or
                        (self.hdf5_data is not None and self.hdf5_step_idx >= self.total_steps)):
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
            cv2.destroyAllWindows()
            self.cleanup()
        
        rospy.loginfo("Diffusion policy execution stopped")
    
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
    
    def load_hdf5_data(self):
        """Load observation data from HDF5 file"""
        try:
            with h5py.File(self.hdf5_data_path, 'r') as f:
                # Load all data into memory for faster access
                demo_key = list(f['data'].keys())[0]  # Use first demo
                self.hdf5_data = {
                    'agentview_image': f[f'data/{demo_key}/obs/agentview_image'][()],
                    'eef_pos': f[f'data/{demo_key}/obs/eef_pos'][()],
                    'eef_rot': f[f'data/{demo_key}/obs/eef_rot'][()],
                    'gripper_qpos': f[f'data/{demo_key}/obs/gripper_qpos'][()],
                    'arm_qpos': f[f'data/{demo_key}/obs/arm_qpos'][()],
                    'time': f[f'data/{demo_key}/obs/time'][()]
                }
                self.total_steps = len(self.hdf5_data['agentview_image'])
                print(f"Loaded HDF5 data with {self.total_steps} steps from {self.hdf5_data_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load HDF5 data: {e}")
            self.hdf5_data = None
    
    def get_next_hdf5_sample(self):
        """Get the next sample from HDF5 data"""
        if self.hdf5_data is None:
            return None
        
        # Loop until we find a valid sample or reach the end
        for _ in range(100):  # Avoid infinite loop, limit to 100 attempts
            # Get joint states and image
            joint_states = self.hdf5_joint_states[self.hdf5_step_idx]
            image = self.hdf5_images[self.hdf5_step_idx]
            
            # Increment index
            self.hdf5_step_idx += 1
            if self.hdf5_step_idx >= len(self.hdf5_joint_states):
                self.hdf5_step_idx = 0  # Loop back to start
            
            # Check if image is valid
            if image is not None and np.prod(image.shape) > 0:
                return joint_states, image
        
        return None
    
    def run_with_hdf5(self):
        """Run the policy execution with HDF5 data"""
        rospy.loginfo("Starting diffusion policy execution with HDF5 data...")
        
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

                    # Get next sample from HDF5 data
                    hdf5_sample = self.get_next_hdf5_sample()
                    if hdf5_sample is not None:
                        self.current_joint_states, color_image = hdf5_sample
                        
                        # TODO: Handle depth image if available in HDF5
                        # self.current_depth_image = ...
                        
                        # Predict action chunk
                        action_chunk = self.predict_action()
                        
                        if action_chunk is not None:
                            # Execute action chunk
                            self.execute_action_chunk(action_chunk)
                        else:
                            rospy.logwarn_throttle(5, "No action predicted, waiting for valid input...")
                    else:
                        rospy.logwarn_throttle(5, "No valid HDF5 sample, skipping...")
                    
                    rate.sleep()
                    
                except Exception as e:
                    rospy.logerr(f"Error in main loop with HDF5: {e}")
                    continue
                    
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt received")
        finally:
            self.cleanup()
        
        rospy.loginfo("Diffusion policy execution with HDF5 data stopped")
    
    def get_ee_pose_from_hdf5_or_pybullet(self):
        """Get end-effector pose from HDF5 data or PyBullet simulation"""
        # if self.hdf5_data is not None:
        #     # Read from HDF5 file
        #     if self.hdf5_step_idx > 0:  # Use previous step since we already incremented in get_camera_image
        #         step_idx = self.hdf5_step_idx - 1
        #     else:
        #         step_idx = 0
                
        #     ee_pos = self.hdf5_data['eef_pos'][step_idx]
        #     ee_quat = self.hdf5_data['eef_rot'][step_idx]
            
        #     # Convert quaternion to rotation matrix
        #     ee_rot = Rotation.from_quat(ee_quat).as_matrix()
            
        #     return ee_pos, ee_rot
        # else:
        #     # Use PyBullet simulation (original implementation)
        #     return self.get_ee_pose_from_pybullet()
        return self.get_ee_pose_from_pybullet()

    def get_current_states_from_hdf5_or_hardware(self):
        """Get current robot states from HDF5 data or hardware"""
        if self.hdf5_data is not None:
            # Read from HDF5 file
            if self.hdf5_step_idx > 0:
                step_idx = self.hdf5_step_idx - 1
            else:
                step_idx = 0
                
            # Get arm joint states
            arm_qpos = self.hdf5_data['arm_qpos'][step_idx]
            
            # Get gripper states
            gripper_qpos = self.hdf5_data['gripper_qpos'][step_idx]
            
            return arm_qpos, gripper_qpos
        else:
            # Use hardware readings (original implementation)
            arm_qpos = self.kinova_node.get_current_joint_angles() if hasattr(self, 'kinova_node') else None
            try:
                gripper_qpos = self.leap_node.read_pos() if hasattr(self, 'leap_node') else None
            except:
                gripper_qpos = None
            return arm_qpos, gripper_qpos
    


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy diffusion policy on real robot')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to diffusion policy checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on')
    parser.add_argument('--hdf5_data', type=str, 
                        default='/media/yaxun/B197/teleop_data/50-in-vision/leap_action_0.hdf5',
                        help='Path to HDF5 file for observation data')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run the system
        system = RealSystemDiffusionPolicy(args.checkpoint, args.device, args.hdf5_data)
        system.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down...")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")


if __name__ == '__main__':
    main()