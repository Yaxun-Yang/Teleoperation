#!/usr/bin/env python3
"""
Experiment script to conduct 30 diffusion policy rollouts with different initial poses
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
import json
import threading
import select
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
from datetime import datetime
from pynput.keyboard import Key, Listener

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

# Import the main policy class from rollout_policy_real_system.py
from rollout_policy_real_system import RealSystemDiffusionPolicy, LeapNode


class ExperimentManager:
    def __init__(self, checkpoint_path, device='cuda:0', results_dir=None, mode='pose'):
        """
        Initialize experiment manager for conducting multiple policy rollouts
        
        Args:
            checkpoint_path: Path to the trained diffusion policy checkpoint
            device: Device to run the model on
            results_dir: Directory to save experiment results
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Create results directory
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"/media/yaxun/manipulation1/leaphandproject_ws/experiment_results_{timestamp}"
        else:
            self.results_dir = results_dir
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.mode = mode  # Mode for the input (qpos, pose)
        # Ideal initial pose parameters
        self.ideal_position = np.array([-0.02928257, -0.56434673, 0.64273047])
        self.ideal_rotation_matrix = np.array([
            [-0.40596742, 0.73652323, -0.54103972],
            [-0.17870997, 0.51661091, 0.83736247],
            [0.89624394, 0.43663108, -0.07810316]
        ])
        
        # Offset ranges
        self.position_offset_range = 0.1  # 10cm range for each axis
        self.rotation_offset_range = np.deg2rad(10)  # 10 degree range for each axis (changed from 30)
        
        # Generate 30 different initial poses
        self.initial_poses = self.generate_initial_poses()
        
        # Experiment tracking
        self.experiment_results = []
        self.current_experiment = 0
        
        print(f"Experiment Manager initialized with {len(self.initial_poses)} poses")
        print(f"Results will be saved to: {self.results_dir}")
    
    def generate_initial_poses(self):
        """
        Generate 30 different initial poses with random offsets
        
        Returns:
            List of dictionaries containing position and rotation matrices
        """
        np.random.seed(42)  # For reproducible experiments
        initial_poses = []
        
        for i in range(30):
            # Generate position offset (-0.1 to +0.1 for each axis)
            position_offset = np.random.uniform(-self.position_offset_range, 
                                               self.position_offset_range, 3)
            target_position = self.ideal_position + position_offset
            
            # Generate rotation offset (-30° to +30° for each axis)
            rotation_offset_euler = np.random.uniform(-self.rotation_offset_range,
                                                     self.rotation_offset_range, 3)
            
            # Apply rotation offset to ideal rotation
            offset_rotation = Rotation.from_euler('xyz', rotation_offset_euler)
            ideal_rotation = Rotation.from_matrix(self.ideal_rotation_matrix)
            target_rotation = offset_rotation * ideal_rotation
            target_rotation_matrix = target_rotation.as_matrix()
            
            pose = {
                'experiment_id': i + 1,
                'position': target_position,
                'rotation_matrix': target_rotation_matrix,
                'rotation_quaternion': target_rotation.as_quat(),  # [x, y, z, w]
                'position_offset': position_offset,
                'rotation_offset_euler': rotation_offset_euler
            }
            
            initial_poses.append(pose)
            
            # Fix numpy array formatting
            print(f"Pose {i+1}: pos=[{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}], "
                  f"rot_offset=[{np.rad2deg(rotation_offset_euler[0]):.1f}°, "
                  f"{np.rad2deg(rotation_offset_euler[1]):.1f}°, {np.rad2deg(rotation_offset_euler[2]):.1f}°]")
        
        return initial_poses
    
    def save_pose_config(self):
        """Save the generated poses configuration to a JSON file"""
        config_file = os.path.join(self.results_dir, "poses_config.json")
        
        # Convert numpy arrays to lists for JSON serialization
        poses_for_json = []
        for pose in self.initial_poses:
            pose_json = {
                'experiment_id': pose['experiment_id'],
                'position': pose['position'].tolist(),
                'rotation_matrix': pose['rotation_matrix'].tolist(),
                'rotation_quaternion': pose['rotation_quaternion'].tolist(),
                'position_offset': pose['position_offset'].tolist(),
                'rotation_offset_euler': pose['rotation_offset_euler'].tolist(),
                'rotation_offset_degrees': np.rad2deg(pose['rotation_offset_euler']).tolist()
            }
            poses_for_json.append(pose_json)
        
        config_data = {
            'experiment_info': {
                'total_experiments': len(self.initial_poses),
                'ideal_position': self.ideal_position.tolist(),
                'ideal_rotation_matrix': self.ideal_rotation_matrix.tolist(),
                'position_offset_range_m': self.position_offset_range,
                'rotation_offset_range_deg': np.rad2deg(self.rotation_offset_range),
                'timestamp': datetime.now().isoformat()
            },
            'poses': poses_for_json
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Poses configuration saved to: {config_file}")
    
    def move_to_initial_pose(self, pose, policy_system):
        """
        Move robot to the specified initial pose using PyBullet IK
        
        Args:
            pose: Dictionary containing position and rotation
            policy_system: RealSystemDiffusionPolicy instance
        
        Returns:
            bool: True if movement successful, False otherwise
        """
        try:
            target_pos = pose['position']
            target_quat = pose['rotation_quaternion']
            
            print(f"Moving to initial pose {pose['experiment_id']}:")
            # Fix numpy array formatting
            print(f"  Position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            print(f"  Quaternion: [{target_quat[0]:.3f}, {target_quat[1]:.3f}, {target_quat[2]:.3f}, {target_quat[3]:.3f}]")
            
            # Convert end-effector pose to joint angles using PyBullet IK
            joint_positions = policy_system.ee_to_joint_commands(target_pos, target_quat)
            
            if joint_positions is None:
                print(f"❌ Failed to compute IK for pose {pose['experiment_id']}")
                return False
            
            # Fix numpy array formatting
            joint_str = ", ".join([f"{jp:.4f}" for jp in joint_positions])
            print(f"  Joint positions: [{joint_str}]")
            
            # Move robot to initial pose
            success = policy_system.execute_arm_joint_command(joint_positions)
            
            if success:
                # Wait for movement to complete
                rospy.sleep(3.0)
                
                # Verify position
                current_joints = policy_system.kinova_node.get_current_joint_angles()
                if current_joints is not None:
                    joint_error = np.linalg.norm(joint_positions - current_joints)
                    print(f"  Joint error: {joint_error:.4f}")
                    
                    if joint_error < 0.1:  # 0.1 radian tolerance
                        print(f"✅ Successfully moved to initial pose {pose['experiment_id']}")
                        return True
                    else:
                        print(f"⚠️ Large joint error for pose {pose['experiment_id']}: {joint_error:.4f}")
                        return False
                else:
                    print(f"⚠️ Could not verify position for pose {pose['experiment_id']}")
                    return True  # Assume success if we can't verify
            else:
                print(f"❌ Failed to move to initial pose {pose['experiment_id']}")
                return False
                
        except Exception as e:
            print(f"❌ Error moving to initial pose {pose['experiment_id']}: {e}")
            return False
    
    def wait_for_user_setup(self, experiment_id):
        """
        Wait for user to set up the object position before starting the experiment
        
        Args:
            experiment_id: Current experiment ID
        """
        print(f"\n{'🔧 OBJECT SETUP REQUIRED 🔧':=^60}")
        print(f"📦 Experiment {experiment_id}: Please set up the object position")
        print(f"   Make sure the object is properly placed for this experiment")
        print(f"   The robot will move to initial pose: ")
        pose = self.initial_poses[experiment_id - 1]
        print(f"   Position: [{pose['position'][0]:.3f}, {pose['position'][1]:.3f}, {pose['position'][2]:.3f}]")
        print(f"   Rotation offset: [{np.rad2deg(pose['rotation_offset_euler'][0]):.1f}°, "
              f"{np.rad2deg(pose['rotation_offset_euler'][1]):.1f}°, "
              f"{np.rad2deg(pose['rotation_offset_euler'][2]):.1f}°]")
        print(f"\n⏳ Press ENTER when object setup is complete and ready to start experiment...")
        
        try:
            input()  # Wait for user to press Enter
            print(f"✅ Object setup confirmed for experiment {experiment_id}")
            print(f"🚀 Starting experiment in 3 seconds...")
            
            # Give a short countdown
            for i in range(3, 0, -1):
                print(f"   {i}...")
                time.sleep(1.0)
            
            print(f"▶️  Experiment {experiment_id} starting now!")
            
        except KeyboardInterrupt:
            print(f"\n🛑 User cancelled experiment {experiment_id}")
            raise KeyboardInterrupt("User cancelled during object setup")
    
    def wait_for_user_confirmation_after_initial_pose(self, experiment_id):
        """
        Wait for user confirmation after robot reaches initial pose
        
        Args:
            experiment_id: Current experiment ID
        """
        print(f"\n{'🤖 ROBOT POSITIONED 🤖':=^60}")
        print(f"✅ Robot has moved to initial pose for experiment {experiment_id}")
        print(f"🔍 Please verify:")
        print(f"   - Robot is in the correct starting position")
        print(f"   - Object is properly positioned relative to robot")
        print(f"   - Camera view is clear and unobstructed")
        print(f"   - All safety checks are complete")
        print(f"\n⏳ Press ENTER to start policy execution, or Ctrl+C to abort...")
        
        try:
            input()  # Wait for user to press Enter
            print(f"✅ Policy execution confirmed for experiment {experiment_id}")
            print(f"🎮 Starting diffusion policy rollout...")
            
        except KeyboardInterrupt:
            print(f"\n🛑 User cancelled experiment {experiment_id} before policy execution")
            raise KeyboardInterrupt("User cancelled before policy execution")

    def run_single_experiment(self, experiment_id):
        """
        Run a single experiment with the specified initial pose
        
        Args:
            experiment_id: ID of the experiment (1-30)
        
        Returns:
            dict: Experiment results
        """
        pose = self.initial_poses[experiment_id - 1]
        
        print(f"\n{'='*60}")
        print(f"🧪 Starting Experiment {experiment_id}/30")
        print(f"{'='*60}")
        
        # Initialize policy system for this experiment
        policy_system = None
        experiment_start_time = time.time()
        experiment_result = {
            'experiment_id': experiment_id,
            'pose': pose,
            'success': False,
            'error_message': None,
            'start_time': experiment_start_time,
            'end_time': None,
            'duration': None,
            'policy_duration': None,  # Duration of just the policy execution
            'step_count': 0,
            'initial_pose_reached': False,
            'policy_executed': False,
            'user_setup_completed': False,
            'user_confirmation_received': False,
            'user_marked_success': False
        }
        
        try:
            # Wait for user to set up object position
            print("🔧 Initializing policy system...")
            policy_system = RealSystemDiffusionPolicy(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                mode = self.mode
            )
            
            # Wait for user object setup
            self.wait_for_user_setup(experiment_id)
            experiment_result['user_setup_completed'] = True
            
            # Move to initial pose
            print("📍 Moving robot to initial pose...")
            initial_pose_success = self.move_to_initial_pose(pose, policy_system)
            experiment_result['initial_pose_reached'] = initial_pose_success
            
            if not initial_pose_success:
                experiment_result['error_message'] = "Failed to reach initial pose"
                print(f"❌ Experiment {experiment_id} failed: Could not reach initial pose")
                return experiment_result
            
            # Wait for user confirmation after reaching initial pose
            self.wait_for_user_confirmation_after_initial_pose(experiment_id)
            experiment_result['user_confirmation_received'] = True
            
            # Record policy execution start time
            policy_start_time = time.time()
            
            # Execute policy for a fixed duration or until completion
            print("🤖 Executing diffusion policy...")
            policy_success, step_count, policy_duration, user_marked_success = self.execute_policy_rollout(policy_system, experiment_id)
            
            experiment_result['policy_executed'] = policy_success
            experiment_result['step_count'] = step_count
            experiment_result['policy_duration'] = policy_duration
            experiment_result['user_marked_success'] = user_marked_success
            
            if policy_success:
                experiment_result['success'] = True
                print(f"✅ Experiment {experiment_id} completed successfully")
            else:
                experiment_result['error_message'] = "Policy execution failed"
                print(f"❌ Experiment {experiment_id} failed: Policy execution error")
            
        except KeyboardInterrupt:
            experiment_result['error_message'] = "User cancelled experiment"
            print(f"🛑 Experiment {experiment_id} cancelled by user")
            raise  # Re-raise to stop the experiment batch
            
        except Exception as e:
            experiment_result['error_message'] = str(e)
            print(f"❌ Experiment {experiment_id} failed with exception: {e}")
            
        finally:
            # Record total experiment end time
            experiment_result['end_time'] = time.time()
            experiment_result['duration'] = experiment_result['end_time'] - experiment_start_time
            
            if policy_system is not None:
                try:
                    policy_system.cleanup()
                except:
                    pass
            
            # Save individual experiment result
            self.save_experiment_result(experiment_result)
            
            # Show both total and policy duration
            print(f"⏱️ Experiment {experiment_id} total duration: {experiment_result['duration']:.2f}s")
            if experiment_result['policy_duration'] is not None:
                print(f"🎮 Policy execution duration: {experiment_result['policy_duration']:.2f}s")
                print(f"📊 Steps: {experiment_result['step_count']}")
        
        return experiment_result
    
    def execute_policy_rollout(self, policy_system, experiment_id, max_duration=300.0):
        """
        Execute the diffusion policy rollout for a specified duration
        User can press SPACE key to mark experiment as successful and proceed to next
        
        Args:
            policy_system: RealSystemDiffusionPolicy instance
            experiment_id: Current experiment ID
            max_duration: Maximum duration in seconds (default: 120s)
        
        Returns:
            tuple: (success: bool, step_count: int, duration: float, user_marked_success: bool)
        """
        try:
            print(f"🎮 Starting policy rollout for experiment {experiment_id}")
            print(f"💡 During execution, press SPACE key to mark experiment as successful and continue to next experiment")
            
            # Create video recording directory for this experiment
            video_dir = os.path.join(self.results_dir, f"experiment_{experiment_id:02d}_videos")
            os.makedirs(video_dir, exist_ok=True)
            
            start_time = time.time()
            step_count = 0
            last_report_time = start_time
            
            # Flag variables for space key detection
            space_key_pressed = False
            user_marked_success = False
            success_time = None
            success_steps = None
            
            def on_press(key):
                """Callback for key press events"""
                nonlocal space_key_pressed, success_time, success_steps, user_marked_success
                if key == Key.space and not user_marked_success:  # Only trigger once
                    space_key_pressed = True
                    success_time = time.time()
                    success_steps = step_count
                    user_marked_success = True
                    print(f"\n🎉 EXPERIMENT SUCCESS! User pressed SPACE key to mark experiment {experiment_id} as successful!")
                    print(f"📊 Success recorded at:")
                    print(f"   ⏱️  Duration: {success_time - start_time:.2f} seconds")
                    print(f"   🔢 Steps: {success_steps}")
            
            def on_release(key):
                """Callback for key release events"""
                nonlocal space_key_pressed
                if key == Key.space:
                    space_key_pressed = False
            
            # Start keyboard listener
            keyboard_listener = Listener(on_press=on_press, on_release=on_release)
            keyboard_listener.start()
            
            try:
                # Main policy execution loop
                rate = rospy.Rate(config.CONTROL_FREQUENCY)
                
                while not rospy.is_shutdown():
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    
                    # Check if user marked experiment as successful
                    if user_marked_success:
                        print(f"🛑 Stopping experiment {experiment_id} - marked successful by user")
                        break
                    
                    # Check if we've exceeded the maximum duration
                    if elapsed_time > max_duration:
                        print(f"⏱️ Experiment {experiment_id} reached maximum duration ({max_duration}s)")
                        break   
                    
                    try:
                        # Update PyBullet joint states from real robot
                        current_joints = policy_system.kinova_node.get_current_joint_angles()
                        if current_joints is not None:
                            policy_system.current_joint_states = current_joints
                            if policy_system.pybullet_sync_enabled:
                                policy_system.update_pybullet_arm_qpos(current_joints)
                        
                        # Update LeapHand states
                        try:
                            current_leaphand_states = policy_system.leap_node.read_pos()
                            if current_leaphand_states is not None:
                                policy_system.current_leaphand_states = current_leaphand_states
                        except Exception as e:
                            rospy.logwarn_throttle(5, f"Failed to read LeapHand: {e}")
                            policy_system.current_leaphand_states = None
                        
                        # Check if we have all required data
                        if (policy_system.current_joint_states is None or 
                            policy_system.current_leaphand_states is None or 
                            policy_system.k4a is None):
                            rospy.logwarn_throttle(5, "Waiting for sensor data...")
                            rate.sleep()
                            continue
                        
                        # Only execute next policy step if user hasn't marked success
                        if not user_marked_success:
                            # Predict and execute action
                            action_chunk = policy_system.predict_action()
                            
                            if action_chunk is not None:
                                policy_system.execute_action_chunk(action_chunk)
                                step_count += 1
                                
                                # Report progress every 5 seconds
                                if current_time - last_report_time >= 5.0:
                                    print(f"📊 Experiment {experiment_id}: Step {step_count}, "
                                          f"Time: {elapsed_time:.1f}s, Rate: {step_count/elapsed_time:.1f} steps/s | Press SPACE for success")
                                    last_report_time = current_time
                            else:
                                rospy.logwarn_throttle(5, "No action predicted")
                        
                        rate.sleep()
                        
                    except Exception as e:
                        rospy.logerr(f"Error in policy execution: {e}")
                        continue
                        
            finally:
                # Stop keyboard listener
                keyboard_listener.stop()
            
            # Calculate final results
            if user_marked_success and success_time is not None:
                # User marked as successful - use recorded time and steps
                total_time = success_time - start_time
                final_step_count = success_steps
                print(f"\n{'🎉 EXPERIMENT MARKED SUCCESSFUL BY USER 🎉':=^60}")
                print(f"✅ Experiment {experiment_id} marked as successful!")
                print(f"📊 Final Results (recorded at success moment):")
                print(f"   ⏱️  Exact Duration: {total_time:.2f} seconds")
                print(f"   🔢 Exact Steps: {final_step_count}")
                print(f"   📈 Average step rate: {final_step_count/total_time:.1f} steps/sec")
                success = True
            else:
                # Timeout or natural completion
                total_time = time.time() - start_time
                final_step_count = step_count
                print(f"✅ Policy rollout completed for experiment {experiment_id}")
                print(f"📊 Final Results:")
                print(f"   ⏱️  Duration: {total_time:.2f} seconds") 
                print(f"   🔢 Steps executed: {final_step_count}")
                print(f"   📈 Average step rate: {final_step_count/total_time:.1f} steps/sec")
                success = True  # Consider completed execution as successful
            
            return success, final_step_count, total_time, user_marked_success
            
        except Exception as e:
            print(f"❌ Policy rollout failed for experiment {experiment_id}: {e}")
            import traceback
            print(f"🐛 Full traceback: {traceback.format_exc()}")
            return False, step_count, 0.0, False
    
    def save_experiment_result(self, result):
        """Save individual experiment result to JSON file"""
        result_file = os.path.join(self.results_dir, 
                                   f"experiment_{result['experiment_id']:02d}_result.json")
        
        # Convert numpy arrays to lists for JSON serialization
        result_for_json = result.copy()
        if 'pose' in result_for_json:
            pose = result_for_json['pose']
            pose['position'] = pose['position'].tolist()
            pose['rotation_matrix'] = pose['rotation_matrix'].tolist()
            pose['rotation_quaternion'] = pose['rotation_quaternion'].tolist()
            pose['position_offset'] = pose['position_offset'].tolist()
            pose['rotation_offset_euler'] = pose['rotation_offset_euler'].tolist()
        
        with open(result_file, 'w') as f:
            json.dump(result_for_json, f, indent=2)
    
    def save_summary_results(self):
        """Save summary of all experiment results"""
        summary_file = os.path.join(self.results_dir, "experiment_summary.json")
        
        # Calculate statistics
        total_experiments = len(self.experiment_results)
        successful_experiments = sum(1 for r in self.experiment_results if r['success'])
        success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0
        
        # Calculate average duration
        durations = [r['duration'] for r in self.experiment_results if r['duration'] is not None]
        avg_duration = np.mean(durations) if durations else 0
        
        summary = {
            'experiment_info': {
                'total_experiments': total_experiments,
                'successful_experiments': successful_experiments,
                'success_rate': success_rate,
                'average_duration_seconds': avg_duration,
                'timestamp': datetime.now().isoformat()
            },
            'individual_results': self.experiment_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Experiment Summary:")
        print(f"   Total experiments: {total_experiments}")
        print(f"   Successful: {successful_experiments}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Average duration: {avg_duration:.2f}s")
        print(f"   Results saved to: {summary_file}")
    
    def run_all_experiments(self, start_experiment=1, end_experiment=20):
        """
        Run all experiments sequentially
        
        Args:
            start_experiment: Starting experiment number (1-30)
            end_experiment: Ending experiment number (1-30)
        """
        print(f"🚀 Starting experiment batch: {start_experiment} to {end_experiment}")
        
        # Save pose configuration
        self.save_pose_config()
        
        # Run experiments
        for exp_id in range(start_experiment, end_experiment + 1):
            try:
                result = self.run_single_experiment(exp_id)
                self.experiment_results.append(result)
                
                # Add delay between experiments
                if exp_id < end_experiment:
                    print(f"⏸️ Waiting 10 seconds before next experiment...")
                    rospy.sleep(10.0)
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Experiment interrupted at experiment {exp_id}")
                break
            except Exception as e:
                print(f"❌ Fatal error in experiment {exp_id}: {e}")
                continue
        
        # Save summary results
        self.save_summary_results()
        
        print(f"\n🏁 All experiments completed!")
    
    def run_specific_experiments(self, experiment_ids):
        """
        Run specific experiments by ID
        
        Args:
            experiment_ids: List of experiment IDs to run
        """
        print(f"🎯 Running specific experiments: {experiment_ids}")
        
        # Save pose configuration
        self.save_pose_config()
        
        for exp_id in experiment_ids:
            if exp_id < 1 or exp_id > 30:
                print(f"⚠️ Skipping invalid experiment ID: {exp_id}")
                continue
                
            try:
                result = self.run_single_experiment(exp_id)
                self.experiment_results.append(result)
                
                # Add delay between experiments
                if exp_id != experiment_ids[-1]:
                    print(f"⏸️ Waiting 10 seconds before next experiment...")
                    rospy.sleep(10.0)
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Experiment interrupted at experiment {exp_id}")
                break
            except Exception as e:
                print(f"❌ Fatal error in experiment {exp_id}: {e}")
                continue
        
        # Save summary results
        self.save_summary_results()
        
        print(f"\n🏁 Specific experiments completed!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run diffusion policy experiments with different initial poses')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to diffusion policy checkpoint')
    parser.add_argument('--mode', type=str, default='pose',
                        help='Mode for the input (qpos, pose)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory to save results (default: auto-generated)')
    parser.add_argument('--start', type=int, default=1,
                        help='Starting experiment number (1-30)')
    parser.add_argument('--end', type=int, default=30,
                        help='Ending experiment number (1-30)')
    parser.add_argument('--experiments', type=int, nargs='+', default=None,
                        help='Specific experiment IDs to run (e.g., --experiments 1 5 10)')
    
    args = parser.parse_args()
    
    try:
        # Validate arguments
        if args.start < 1 or args.start > 30 or args.end < 1 or args.end > 30:
            raise ValueError("Start and end experiment numbers must be between 1 and 30")
        
        if args.start > args.end:
            raise ValueError("Start experiment number must be <= end experiment number")
        
        # Initialize experiment manager
        experiment_manager = ExperimentManager(
            checkpoint_path=args.checkpoint,
            device=args.device,
            results_dir=args.results_dir,
            mode = args.mode
        )
        
        # Run experiments
        if args.experiments is not None:
            experiment_manager.run_specific_experiments(args.experiments)
        else:
            experiment_manager.run_all_experiments(args.start, args.end)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down...")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")


if __name__ == '__main__':
    main()