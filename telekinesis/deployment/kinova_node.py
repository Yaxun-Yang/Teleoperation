#!/usr/bin/env python3
"""
Kinova ROS node interface for real system deployment
"""

import rospy
import actionlib
import kinova_msgs.msg
from sensor_msgs.msg import JointState
import numpy as np
import math

class KinovaNode:
    """Interface for Kinova arm control via ROS"""
    
    def __init__(self):
        # Initialize action client
        action_address = '/j2n6s300_driver/joints_action/joint_angles'
        self.client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmJointAnglesAction)
        self.client.wait_for_server()
        
        # Current joint positions
        self.current_qpos = {}

        # Subscribe to joint states
        self.joint_state_sub = rospy.Subscriber('/j2n6s300_driver/out/joint_state', JointState, self.joint_state_callback)
        

        
        rospy.loginfo("Kinova node initialized")
    
    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        if len(msg.name) >= 6 and len(msg.position) >= 6:
            # Store joint positions in a dictionary
            for i, name in enumerate(msg.name[:6]):
                if i < len(msg.position):
                    self.current_qpos[name] = msg.position[i]
    
    def move_to_joint_angles(self, joint_angles, blocking=True):
        """
        Move arm to specified joint angles
        
        Args:
            joint_angles: Array of 6 joint angles in radians
            blocking: Whether to wait for completion
        """
        if len(joint_angles) != 6:
            rospy.logerr("Expected 6 joint angles, got {}".format(len(joint_angles)))
            return False
        
        try:
            pi = math.pi
            goal = kinova_msgs.msg.ArmJointAnglesGoal()
            goal.angles.joint1 = joint_angles[0] * 360 / (2*pi)
            goal.angles.joint2 = joint_angles[1] * 360 / (2*pi)
            goal.angles.joint3 = joint_angles[2] * 360 / (2*pi)
            goal.angles.joint4 = joint_angles[3] * 360 / (2*pi)
            goal.angles.joint5 = joint_angles[4] * 360 / (2*pi)
            goal.angles.joint6 = joint_angles[5] * 360 / (2*pi)
            
            self.client.send_goal(goal)
            
            if blocking:
                success = self.client.wait_for_result(rospy.Duration(20.0))
                if success:
                    return self.client.get_result()
                else:
                    rospy.logwarn("Joint angle action timed out")
                    self.client.cancel_all_goals()
                    return None
            else:
                return True
                
        except Exception as e:
            rospy.logerr(f"Failed to move to joint angles: {e}")
            return False
    
    def get_current_joint_angles(self):
        """Get current joint angles as a numpy array"""
        if len(self.current_qpos) >= 6:
            # Order joints properly
            joint_names = ['j2n6s300_joint_1', 'j2n6s300_joint_2', 'j2n6s300_joint_3',
                          'j2n6s300_joint_4', 'j2n6s300_joint_5', 'j2n6s300_joint_6']
            
            joint_values = []
            for name in joint_names:
                if name in self.current_qpos:
                    joint_values.append(self.current_qpos[name])
                else:
                    rospy.logwarn(f"Joint {name} not found in current positions")
                    return None
            
            return np.array(joint_values)
        else:
            rospy.logwarn("Insufficient joint state data")
            return None
    
    def cancel_all_goals(self):
        """Cancel all pending goals"""
        self.client.cancel_all_goals()
    
    def is_moving(self):
        """Check if arm is currently moving"""
        state = self.client.get_state()
        return state == actionlib.GoalStatus.ACTIVE or state == actionlib.GoalStatus.PENDING
