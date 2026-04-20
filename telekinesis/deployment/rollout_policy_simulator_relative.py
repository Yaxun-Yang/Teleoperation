#!/usr/bin/env python3
"""
PyBullet Simulator deployment of diffusion policy
Reads observations from HDF5 dataset and executes actions in simulation
"""

import os
import sys
import numpy as np
import torch
import hydra
import dill
import time
import cv2
import h5py
import pybullet as p
import pybullet_data

# Add necessary paths
import path_config
sys.path.append(path_config.DIFFUSION_POLICY_DIR)
sys.path.append(path_config.SRC_DIR)

from diffusion_policy.model.common.rotation_transformer import RotationTransformer
import real_system_config as config


class SimulatorDiffusionPolicy:
    def __init__(self, checkpoint_path, dataset_path, device='cuda:0', mode='pose', demo_idx=0):
        """
        Initialize simulator diffusion policy
        
        Args:
            checkpoint_path: Path to the trained diffusion policy checkpoint
            dataset_path: Path to HDF5 dataset for observations
            device: Device to run the model on
            mode: 'pose' or 'qpos'
            demo_idx: Which demo to use from dataset
        """
        self.device = torch.device(device)
        self.mode = mode
        self.demo_idx = demo_idx
        
        # Load dataset
        self.load_dataset(dataset_path)
        
        # Load diffusion policy model
        self.load_policy(checkpoint_path)
        
        # Initialize PyBullet simulation
        self.init_pybullet()

        # State variables
        self.past_obs = None
        self.second_to_last_obs = None
        self.current_timestep = 0
        
        # Rotation transformers
        self.mat_to_quat = RotationTransformer('matrix', 'quaternion')
        self.quat_to_mat = RotationTransformer('quaternion', 'matrix')
        self.six_to_quat = RotationTransformer('rotation_6d', 'quaternion')
        
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

    
    def load_dataset(self, dataset_path):
        """Load HDF5 dataset"""
        print(f"Loading dataset from {dataset_path}")
        self.dataset = h5py.File(dataset_path, 'r')
        self.demo_key = f'demo_{self.demo_idx}'
        
        # Load observations
        self.obs_data = {
            'agentview_image': self.dataset['data'][self.demo_key]['obs']['agentview_image'][:],
            'arm_qpos': self.dataset['data'][self.demo_key]['obs']['arm_qpos'][:],
            'eef_pos': self.dataset['data'][self.demo_key]['obs']['eef_pos'][:],
            'eef_quat': self.dataset['data'][self.demo_key]['obs']['eef_quat'][:],
            'gripper_qpos': self.dataset['data'][self.demo_key]['obs']['gripper_qpos'][:]
        }
        
        self.num_timesteps = len(self.obs_data['arm_qpos'])
        print(f"Loaded demo {self.demo_idx} with {self.num_timesteps} timesteps")
    
    def init_pybullet(self):
        """Initialize PyBullet simulation"""
        # Connect to PyBullet in GUI mode for visualization
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set up simulation
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Load Kinova robot at position (0.5, 0, 0)
        self.kinova_base_pos = [0.0, 0.0, 0.0]
        self.kinova_id = p.loadURDF(
            config.KINOVA_URDF_PATH,
            self.kinova_base_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        
        # Initialize robot to first pose from dataset
        initial_qpos = self.obs_data['arm_qpos'][0]
        for i, joint_pos in enumerate(initial_qpos):
            p.resetJointState(self.kinova_id, i + 2, joint_pos)
        
        # output joint info
        for i in range(p.getNumJoints(self.kinova_id)):
            print(p.getJointInfo(self.kinova_id, i))
        
        p.stepSimulation()
        
        # Configure camera to view both robots
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.25, 0.0, 0.2]
        )
        
        print(f"PyBullet simulation initialized:")
        print(f"  - Kinova at {self.kinova_base_pos}")
        print(f"  - LeapHand at origin (0, 0, 0.1)")
    
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
    
    def get_ee_pose_from_pybullet(self):
        """Get end-effector pose from PyBullet simulation"""
        try:
            link_state = p.getLinkState(self.kinova_id, config.KINOVA_END_EFFECTOR_INDEX)
            ee_pos = np.array(link_state[4])
            ee_quat = np.array(link_state[5])  # PyBullet format: [x, y, z, w]
            ee_quat = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])  # Convert to [w, x, y, z]
            ee_rot = self.quat_to_mat.forward(torch.tensor(ee_quat, dtype=torch.float32)[None]).squeeze(0).numpy()
            
            return ee_pos, ee_rot
        except Exception as e:
            print(f"Error getting EE pose from PyBullet: {e}")
            return None, None
    
    def preprocess_image(self, image):
        """Preprocess camera image for the model"""
        if image is None:
            return None
        
        # Resize image
        resized = cv2.resize(image, config.IMAGE_SIZE)
        
        # Normalize to [0,1] range
        normalized = torch.tensor(resized).permute(2, 0, 1).float() / 255.0
        return normalized
    
    def get_observation_from_dataset(self, timestep):
        """Get observation from dataset at given timestep"""
        if timestep >= self.num_timesteps:
            return None
        
        obs = {
            'agentview_image': self.preprocess_image(self.obs_data['agentview_image'][timestep]),
            'eef_pos': torch.tensor(self.obs_data['eef_pos'][timestep], dtype=torch.float32),
            'eef_quat': torch.tensor(self.obs_data['eef_quat'][timestep], dtype=torch.float32),
            'gripper_qpos': torch.tensor(self.obs_data['gripper_qpos'][timestep], dtype=torch.float32),
            'arm_qpos': torch.tensor(self.obs_data['arm_qpos'][timestep], dtype=torch.float32)
        }
        return obs
    
    def prepare_model_input(self):
        """Prepare input for the diffusion policy model"""        
        if self.mode == 'pose':
            # Use EEF pose observations
            if self.past_obs is not None:
                model_input = {
                    'agentview_image': torch.stack([
                        self.past_obs['agentview_image'], 
                        self.current_obs['agentview_image']
                    ]),
                    'eef_pos': torch.stack([
                        self.past_obs['eef_pos'], 
                        self.current_obs['eef_pos']
                    ]),
                    'eef_quat': torch.stack([
                        self.past_obs['eef_quat'], 
                        self.current_obs['eef_quat']
                    ]),
                    'gripper_qpos': torch.stack([
                        self.past_obs['gripper_qpos'], 
                        self.current_obs['gripper_qpos']
                    ])
                }
            else:
                # Get current observation from dataset
                current_obs = self.get_observation_from_dataset(self.current_timestep)
                
                if current_obs is None:
                    return None
                # First step, duplicate current observation
                model_input = {
                    'agentview_image': torch.stack([
                        current_obs['agentview_image'], 
                        current_obs['agentview_image']
                    ]),
                    'eef_pos': torch.stack([
                        current_obs['eef_pos'], 
                        current_obs['eef_pos']
                    ]),
                    'eef_quat': torch.stack([
                        current_obs['eef_quat'], 
                        current_obs['eef_quat']
                    ]),
                    'gripper_qpos': torch.stack([
                        current_obs['gripper_qpos'], 
                        current_obs['gripper_qpos']
                    ])
                }

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
    
    def update_kinova_leaphand(self, gripper_qpos):
        """Update LeapHand joints on Kinova robot in simulation
        
        Kinova URDF structure (PyBullet indexing):
        - Joint 0: connect_root_and_world (fixed)
        - Joint 1: j2n6s300_joint_base (fixed)
        - Joint 2-7: j2n6s300_joint_1 to joint_6 (6 DOF arm)
        - Joint 8: arm_to_assembly (fixed)
        - Joint 9: assembly_to_hand (fixed)
        - Joint 10-25: LeapHand joints 0-15 (16 DOF hand)
        
        Map from dataset order to URDF order:
        Dataset: Thumb(0-3), Index(4-7), Middle(8-11), Ring(12-15)
        URDF:    Index(0-3), Middle(4-7), Ring(8-11), Thumb(12-15)
        
        Note: Kinova URDF hand joints need -π offset to match hardware zero position
        """
        try:
            if len(gripper_qpos) == 16:
                # # Remap from dataset order to URDF order
                # urdf_order = np.zeros(16)
                
                # # Dataset thumb (0-3) -> URDF thumb (12-15)
                # urdf_order[12:16] = gripper_qpos[0:4]
                
                # # Dataset index (4-7) -> URDF index (0-3)
                # urdf_order[0:4] = gripper_qpos[4:8]
                
                # # Dataset middle (8-11) -> URDF middle (4-7)
                # urdf_order[4:8] = gripper_qpos[8:12]
                
                # # Dataset ring (12-15) -> URDF ring (8-11)
                # urdf_order[8:12] = gripper_qpos[12:16]
                
                # # In Kinova URDF, hand joints start at PyBullet index 10
                # # (after base, arm, assembly, and hand attachment joints)
                kinova_hand_start_idx = 10
                
                # # Set joint positions for Kinova's LeapHand with -π offset
                # for i in range(16):
                #     joint_idx = kinova_hand_start_idx + i
                #     joint_pos_adjusted = urdf_order[i] - 3.14159  # Apply -π offset
                #     p.resetJointState(self.kinova_id, joint_idx, joint_pos_adjusted)
                
                p.resetJointState(self.kinova_id, 10, gripper_qpos[1] - np.pi)
                p.resetJointState(self.kinova_id, 11, gripper_qpos[0] - np.pi)
                p.resetJointState(self.kinova_id, 12, gripper_qpos[2] - np.pi)
                p.resetJointState(self.kinova_id, 13, gripper_qpos[3] - np.pi)
                p.resetJointState(self.kinova_id, 15, gripper_qpos[5] - np.pi)
                p.resetJointState(self.kinova_id, 16, gripper_qpos[4] - np.pi)
                p.resetJointState(self.kinova_id, 17, gripper_qpos[6] - np.pi)
                p.resetJointState(self.kinova_id, 18, gripper_qpos[7] - np.pi)
                p.resetJointState(self.kinova_id, 20, gripper_qpos[9] - np.pi)
                p.resetJointState(self.kinova_id, 21, gripper_qpos[8] - np.pi)
                p.resetJointState(self.kinova_id, 22, gripper_qpos[10] - np.pi)
                p.resetJointState(self.kinova_id, 23, gripper_qpos[11] - np.pi)
                p.resetJointState(self.kinova_id, 25, gripper_qpos[12] - np.pi)
                p.resetJointState(self.kinova_id, 26, gripper_qpos[13] - np.pi)
                p.resetJointState(self.kinova_id, 27, gripper_qpos[14] - np.pi)
                p.resetJointState(self.kinova_id, 28, gripper_qpos[15] - np.pi)
                    
                # Debug: print first time to verify
                if not hasattr(self, '_kinova_hand_mapping_printed'):
                    print(f"✓ Kinova LeapHand joint mapping applied:")
                    # print(f"  - Hand joints: PyBullet indices {kinova_hand_start_idx} to {kinova_hand_start_idx + 15}")
                    print(f"  - Offset: -π (-3.14159) applied to all joints")
                    print(f"  - Dataset thumb[0-3] -> URDF[12-15]")
                    print(f"  - Dataset index[4-7] -> URDF[0-3]")
                    print(f"  - Dataset middle[8-11] -> URDF[4-7]")
                    print(f"  - Dataset ring[12-15] -> URDF[8-11]")
                    self._kinova_hand_mapping_printed = True
                    
        except Exception as e:
            print(f"Error updating Kinova LeapHand: {e}")

    def update_leaphand_in_sim(self, gripper_qpos):
        """Update LeapHand joint positions in simulation
        
        Map from dataset order to URDF order:
        Dataset: Thumb(0-3), Index(4-7), Middle(8-11), Ring(12-15)
        URDF:    Index(0-3), Middle(4-7), Ring(8-11), Thumb(12-15)
        """
        # Update standalone LeapHand
        print("gripper_qpos:", gripper_qpos)
        if self.leap_id is not None:
            try:
                # LeapHand has 16 joints
                if len(gripper_qpos) == 16:
                    # Remap from dataset order to URDF order
                    urdf_order = np.zeros(16)
                    
                    # # Dataset thumb (0-3) -> URDF thumb (12-15)
                    # urdf_order[12:16] = gripper_qpos[0:4]
                    
                    # # Dataset index (4-7) -> URDF index (0-3)
                    # urdf_order[0:4] = gripper_qpos[4:8]
                    
                    # # Dataset middle (8-11) -> URDF middle (4-7)
                    # urdf_order[4:8] = gripper_qpos[8:12]
                    
                    # # Dataset ring (12-15) -> URDF ring (8-11)
                    # urdf_order[8:12] = gripper_qpos[12:16]
                    urdf_order = gripper_qpos
                    # Apply offset of -3.14 and set joint positions
                    for i in range(16):
                        joint_pos_adjusted = urdf_order[i] - 3.14
                        p.resetJointState(self.leap_id, i, joint_pos_adjusted)
                        
                    # Debug: print first time to verify mapping
                    if not hasattr(self, '_leap_mapping_printed'):
                        print(f"Standalone LeapHand joint mapping applied:")
                        print(f"  Dataset thumb[0-3] -> URDF[12-15]: {gripper_qpos[0:4]}")
                        print(f"  Dataset index[4-7] -> URDF[0-3]:   {gripper_qpos[4:8]}")
                        print(f"  Dataset middle[8-11] -> URDF[4-7]: {gripper_qpos[8:12]}")
                        print(f"  Dataset ring[12-15] -> URDF[8-11]: {gripper_qpos[12:16]}")
                        self._leap_mapping_printed = True
                        
            except Exception as e:
                print(f"Error updating standalone LeapHand: {e}")
        
        # Update Kinova's LeapHand
        self.update_kinova_leaphand(gripper_qpos)
    
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

    def execute_action_in_sim(self, action, i):
        """Execute a single action in PyBullet simulation"""
        action_np = action.cpu().numpy()
        
        if len(action_np) == 22:
            # Joint space control: [arm_joints(6), leaphand(16)]
            arm_joint_positions = action_np[:6]
            leaphand_positions = action_np[6:22]
            
            # Set arm joint positions
            for i, joint_pos in enumerate(arm_joint_positions):
                p.setJointMotorControl2(
                    self.kinova_id,
                    i + 2,
                    p.POSITION_CONTROL,
                    targetPosition=joint_pos,
                    force=500
                )
            
            # Update LeapHand
            self.update_leaphand_in_sim(leaphand_positions)
            
            print(f"Executed joint action: arm={arm_joint_positions[:3]}..., hand={leaphand_positions[:3]}...")
            
        elif len(action_np) == 25:
            # Task space control: [ee_pos(3), ee_rot_6d(6), leaphand(16)]
            target_pos = action_np[:3]
            target_rot_6d = action_np[3:9]
            leaphand_positions = action_np[9:25]
            print("predict relative pos: ", target_pos)
            # dataset debug, obtain data from dataset
            target_pos = self.get_observation_from_dataset(self.current_timestep+1)['eef_pos']
            target_quat = self.get_observation_from_dataset(self.current_timestep+1)['eef_quat']
            
            # # Convert 6D rotation to quaternion
            target_quat = self.six_to_quat.forward(torch.tensor(target_rot_6d)[None])[0].cpu().numpy()
            target_pos, target_quat = self.convert_relative_to_absolute(target_pos, target_quat)
            target_quat = np.array([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
            
            # PyBullet IK - need to account for Kinova base offset
            joint_positions = p.calculateInverseKinematics(
                self.kinova_id,
                config.KINOVA_END_EFFECTOR_INDEX,
                target_pos,
                target_quat,
                maxNumIterations=100,
                residualThreshold=0.0001
            )
            
            # Set arm joint positions
            for i in range(6):
                p.setJointMotorControl2(
                    self.kinova_id,
                    i + 2,
                    p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=500
                )
            
            # Update LeapHand
            # self.update_leaphand_in_sim(leaphand_positions)
            self.update_kinova_leaphand(leaphand_positions)
            
            print(f"Executed EEF action: pos={target_pos}, quat={target_quat}..., hand={leaphand_positions}...")
        
        # Step simulation
        for _ in range(10):  # Simulate for a few steps
            p.stepSimulation()
            time.sleep(1./240.)
    
    def execute_action_chunk(self, action_chunk):
        """Execute a chunk of actions"""
        num_actions = len(action_chunk)
        # num_actions = 4
        
        for i, action in enumerate(action_chunk):
            # if i == 0:
            #     continue
            self.execute_action_in_sim(action, i)
            
            # Update timestep for observation
            self.current_timestep += 1

            # Record observations for next inference
            self.rollout_step += 1
            # update self.previous_mat
            if self.rollout_step % 8 == 0:
                ee_pos, ee_rot = self.get_ee_pose_from_pybullet()
                self.previous_mat = np.eye(4)
                self.previous_mat[:3, :3] = ee_rot
                self.previous_mat[:3, 3] = ee_pos

            if i == num_actions - 2:
                # Second to last action
                print("current timestep of second to last action: ", self.current_timestep)
                obs = self.get_observation_from_dataset(self.current_timestep)
                if obs is not None:
                    self.past_obs = obs
            
            elif i == num_actions - 1:
                # last action
                print("current timestep of last action: ", self.current_timestep)
                obs = self.get_observation_from_dataset(self.current_timestep)
                if obs is not None:
                    self.current_obs = obs

        return True
    
    def run(self, max_steps=None):
        """Main execution loop"""
        print("Starting simulator diffusion policy execution...")
        
        if max_steps is None:
            max_steps = self.num_timesteps
        
        step_count = 0
        
        try:
            while self.current_timestep < self.num_timesteps and step_count < max_steps:
                print(f"\n=== Step {step_count}, Timestep {self.current_timestep} ===")
                
                # Predict action chunk
                action_chunk = self.predict_action()
                
                if action_chunk is not None:
                    # Execute action chunk
                    self.execute_action_chunk(action_chunk)
                    step_count += 1
                else:
                    print("No action predicted")
                    break
                
                # Small delay for visualization
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        finally:
            self.cleanup()
        
        print("Simulator execution stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Close dataset
            if hasattr(self, 'dataset'):
                self.dataset.close()
            
            # Disconnect PyBullet
            p.disconnect()
            
            print("Cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy diffusion policy in PyBullet simulator')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to diffusion policy checkpoint')
    parser.add_argument('--dataset', type=str, 
                        default='/media/yaxun/B197/teleop_data/output/1104_internalcamera.hdf5',
                        help='Path to HDF5 dataset')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on')
    parser.add_argument('--mode', type=str, default='pose',
                        choices=['pose', 'qpos'],
                        help='Control mode: pose or qpos')
    parser.add_argument('--demo', type=int, default=0,
                        help='Demo index to use from dataset')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum number of steps to run')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run the simulator
        system = SimulatorDiffusionPolicy(
            args.checkpoint, 
            args.dataset,
            args.device,
            args.mode,
            args.demo
        )
        system.run(args.max_steps)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()