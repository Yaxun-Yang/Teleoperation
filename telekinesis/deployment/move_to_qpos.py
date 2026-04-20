#!/usr/bin/env python3
"""
Move Kinova robot to target joint positions using action server
"""

import rospy
import actionlib
import numpy as np
import kinova_msgs.msg
from sensor_msgs.msg import JointState
import argparse
import sys


class KinovaQposController:
    """Controller to move Kinova robot to target joint positions"""
    
    def __init__(self):
        """Initialize the Kinova joint position controller"""
        rospy.init_node('kinova_qpos_controller', anonymous=True)
        
        # Initialize action client
        action_address = '/j2n6s300_driver/joints_action/joint_angles'
        self.client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmJointAnglesAction)
        
        rospy.loginfo("Waiting for action server...")
        self.client.wait_for_server()
        rospy.loginfo("Action server connected!")
        
        # Current joint positions
        self.current_qpos = None
        
        # Subscribe to joint states
        self.joint_state_sub = rospy.Subscriber(
            '/j2n6s300_driver/out/joint_state', 
            JointState, 
            self.joint_state_callback
        )
        
        # Wait for initial joint state
        rospy.loginfo("Waiting for initial joint state...")
        while self.current_qpos is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Initial joint state received!")
        
    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        if len(msg.name) >= 6 and len(msg.position) >= 6:
            # Extract joint positions for the 6 arm joints
            joint_names = ['j2n6s300_joint_1', 'j2n6s300_joint_2', 'j2n6s300_joint_3',
                          'j2n6s300_joint_4', 'j2n6s300_joint_5', 'j2n6s300_joint_6']
            
            joint_values = []
            for name in joint_names:
                if name in msg.name:
                    idx = msg.name.index(name)
                    joint_values.append(msg.position[idx])
                else:
                    rospy.logwarn(f"Joint {name} not found in joint state message")
                    return
            
            self.current_qpos = np.array(joint_values)
    
    def get_current_qpos(self):
        """Get current joint positions"""
        return self.current_qpos.copy() if self.current_qpos is not None else None
    
    def move_to_qpos(self, target_qpos, blocking=True, timeout=30.0):
        """
        Move robot to target joint positions
        
        Args:
            target_qpos: Array of 6 joint angles in radians
            blocking: Whether to wait for completion
            timeout: Timeout in seconds for blocking mode
            
        Returns:
            bool: True if successful, False otherwise
        """
        if len(target_qpos) != 6:
            rospy.logerr(f"Expected 6 joint angles, got {len(target_qpos)}")
            return False
        
        # Get current position for comparison
        current_qpos = self.get_current_qpos()
        if current_qpos is not None:
            # Calculate joint differences
            joint_diff = np.array(target_qpos) - current_qpos
            joint_diff_wrapped = np.arctan2(np.sin(joint_diff), np.cos(joint_diff))
            joint_diff_norm = np.linalg.norm(joint_diff_wrapped)
            
            print(f"\n=== Moving to Target Joint Positions ===")
            print(f"Current qpos (rad): {current_qpos}")
            print(f"Current qpos (deg): {np.degrees(current_qpos)}")
            print(f"Target qpos (rad):  {target_qpos}")
            print(f"Target qpos (deg):  {np.degrees(target_qpos)}")
            print(f"Joint differences (rad): {joint_diff_wrapped}")
            print(f"Joint differences (deg): {np.degrees(joint_diff_wrapped)}")
            print(f"Movement magnitude: {joint_diff_norm:.4f} rad")
            print("=" * 45)
        
        try:
            # Create goal message
            goal = kinova_msgs.msg.ArmJointAnglesGoal()
            goal.angles.joint1 = float(target_qpos[0])
            goal.angles.joint2 = float(target_qpos[1])
            goal.angles.joint3 = float(target_qpos[2])
            goal.angles.joint4 = float(target_qpos[3])
            goal.angles.joint5 = float(target_qpos[4])
            goal.angles.joint6 = float(target_qpos[5])
            
            # Send goal
            rospy.loginfo("Sending joint angle goal...")
            self.client.send_goal(goal)
            
            if blocking:
                # Wait for result
                rospy.loginfo(f"Waiting for motion to complete (timeout: {timeout}s)...")
                success = self.client.wait_for_result(rospy.Duration(timeout))
                
                if success:
                    result = self.client.get_result()
                    rospy.loginfo("Motion completed successfully!")
                    
                    # Show final position
                    final_qpos = self.get_current_qpos()
                    if final_qpos is not None:
                        final_diff = np.array(target_qpos) - final_qpos
                        final_diff_wrapped = np.arctan2(np.sin(final_diff), np.cos(final_diff))
                        final_error = np.linalg.norm(final_diff_wrapped)
                        
                        print(f"\n=== Motion Completed ===")
                        print(f"Final qpos (rad):     {final_qpos}")
                        print(f"Final qpos (deg):     {np.degrees(final_qpos)}")
                        print(f"Target qpos (rad):    {target_qpos}")
                        print(f"Final error (rad):    {final_diff_wrapped}")
                        print(f"Final error (deg):    {np.degrees(final_diff_wrapped)}")
                        print(f"Final error norm:     {final_error:.4f} rad")
                        print("=" * 25)
                    
                    return True
                else:
                    rospy.logwarn("Joint angle action timed out")
                    self.client.cancel_all_goals()
                    return False
            else:
                rospy.loginfo("Goal sent, not waiting for completion")
                return True
                
        except Exception as e:
            rospy.logerr(f"Failed to move to joint angles: {e}")
            return False
    
    def move_relative(self, delta_qpos, blocking=True, timeout=30.0):
        """
        Move robot by relative joint angles
        
        Args:
            delta_qpos: Array of 6 relative joint angle changes in radians
            blocking: Whether to wait for completion
            timeout: Timeout in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        current_qpos = self.get_current_qpos()
        if current_qpos is None:
            rospy.logerr("Cannot get current joint positions")
            return False
        
        target_qpos = current_qpos + np.array(delta_qpos)
        return self.move_to_qpos(target_qpos, blocking, timeout)
    
    def get_joint_limits(self):
        """Get approximate joint limits for Kinova j2n6s300"""
        # Approximate joint limits in radians
        # Note: Check your robot's actual limits in the URDF or documentation
        joint_limits = {
            'lower': np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]),
            'upper': np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
        }
        return joint_limits
    
    def check_joint_limits(self, target_qpos):
        """
        Check if target joint positions are within limits
        
        Args:
            target_qpos: Array of 6 joint angles in radians
            
        Returns:
            bool: True if within limits, False otherwise
        """
        limits = self.get_joint_limits()
        within_limits = np.all((target_qpos >= limits['lower']) & (target_qpos <= limits['upper']))
        
        if not within_limits:
            violations = []
            for i, (pos, low, high) in enumerate(zip(target_qpos, limits['lower'], limits['upper'])):
                if pos < low or pos > high:
                    violations.append(f"Joint {i+1}: {pos:.3f} rad (limits: [{low:.3f}, {high:.3f}])")
            
            rospy.logwarn("Joint limit violations detected:")
            for violation in violations:
                rospy.logwarn(f"  {violation}")
        
        return within_limits
    
    def cancel_motion(self):
        """Cancel current motion"""
        rospy.loginfo("Cancelling current motion...")
        self.client.cancel_all_goals()
    
    def is_moving(self):
        """Check if robot is currently moving"""
        state = self.client.get_state()
        return state == actionlib.GoalStatus.ACTIVE or state == actionlib.GoalStatus.PENDING


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Move Kinova robot to target joint positions')
    parser.add_argument('--qpos', type=float, nargs=6, 
                        help='Target joint positions in radians (6 values)')
    parser.add_argument('--qpos_deg', type=float, nargs=6,
                        help='Target joint positions in degrees (6 values)')
    parser.add_argument('--relative', type=float, nargs=6,
                        help='Relative joint position changes in radians (6 values)')
    parser.add_argument('--relative_deg', type=float, nargs=6,
                        help='Relative joint position changes in degrees (6 values)')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='Timeout in seconds (default: 30.0)')
    parser.add_argument('--no-blocking', action='store_true',
                        help='Do not wait for motion to complete')
    parser.add_argument('--show-current', action='store_true',
                        help='Show current joint positions and exit')
    
    args = parser.parse_args()
    
    try:
        # Initialize controller
        controller = KinovaQposController()
        
        # Show current position if requested
        if args.show_current:
            current_qpos = controller.get_current_qpos()
            if current_qpos is not None:
                print(f"\n=== Current Joint Positions ===")
                print(f"Current qpos (rad): {current_qpos}")
                print(f"Current qpos (deg): {np.degrees(current_qpos)}")
                print("=" * 32)
            else:
                print("Could not get current joint positions")
            return
        
        # Determine target positions
        target_qpos = None
        
        if args.qpos is not None:
            target_qpos = np.array(args.qpos)
            print("Using absolute target positions (radians)")
            
        elif args.qpos_deg is not None:
            target_qpos = np.radians(args.qpos_deg)
            print("Using absolute target positions (converted from degrees)")
            
        elif args.relative is not None:
            delta_qpos = np.array(args.relative)
            print("Using relative position changes (radians)")
            success = controller.move_relative(delta_qpos, not args.no_blocking, args.timeout)
            if success:
                print("Relative motion completed successfully!")
            else:
                print("Relative motion failed!")
            return
            
        elif args.relative_deg is not None:
            delta_qpos = np.radians(args.relative_deg)
            print("Using relative position changes (converted from degrees)")
            success = controller.move_relative(delta_qpos, not args.no_blocking, args.timeout)
            if success:
                print("Relative motion completed successfully!")
            else:
                print("Relative motion failed!")
            return
        
        else:
            print("No target specified. Use --help for usage information.")
            print("\nExample usage:")
            print("  # Move to specific joint angles (radians)")
            print("  python move_to_qpos.py --qpos 0.0 0.5 1.0 0.0 -0.5 0.0")
            print("  # Move to specific joint angles (degrees)")
            print("  python move_to_qpos.py --qpos_deg 0 30 60 0 -30 0")
            print("  # Move relative to current position")
            print("  python move_to_qpos.py --relative 0.1 0.0 0.0 0.0 0.0 0.0")
            print("  # Show current position")
            print("  python move_to_qpos.py --show-current")
            return
        
        # Check joint limits
        if not controller.check_joint_limits(target_qpos):
            response = input("Joint limits violated. Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Motion cancelled due to joint limit violations")
                return
        
        # Execute motion
        success = controller.move_to_qpos(target_qpos, not args.no_blocking, args.timeout)
        
        if success:
            print("Motion completed successfully!")
        else:
            print("Motion failed!")
            
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted by user")
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
        if 'controller' in locals():
            controller.cancel_motion()
    except Exception as e:
        rospy.logerr(f"Error: {e}")


if __name__ == '__main__':
    main()