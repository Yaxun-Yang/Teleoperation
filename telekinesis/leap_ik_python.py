#!/usr/bin/env python3
import pybullet as p
import numpy as np
import os
import time

from oculus_reader.scripts import *
from oculus_reader.scripts.reader import OculusReader

from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
import sys

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
from scipy.spatial.transform import Rotation as R

import keyboard
'''
This takes the glove data, and runs inverse kinematics and then publishes onto LEAP Hand.

Note how the fingertip positions are matching, but the joint angles between the two hands are not.  :) 

Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
'''

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

class LeapNode:
    def __init__(self):
        ####Some parameters
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
           
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
                self.dxl_client.connect()
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #allegro compatibility
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()

class LeapPybulletIK():
    def __init__(self):
        # start pybullet
        #clid = p.connect(p.SHARED_MEMORY)
        #clid = p.connect(p.DIRECT)
        p.connect(p.GUI)
        # load right leap hand      
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        # self.glove_to_leap_mapping_scale = 1.6
        self.glove_to_leap_mapping_scale = 1.6
        # self.leapEndEffectorIndex = [0,1,2,4,5,6,7,9,10,11,12,14,15,16,17,19]
        # self.leapEndEffectorIndex = [0,2,3,4,5,7,8,9,10,12,13,14,15,17,18,19]
        self.leapEndEffectorIndex = [4,9,14,19]

        path_src = os.path.join(path_src, "leap_hand_mesh_right/robot_pybullet.urdf")
        ##You may have to set this path for your setup on ROS2
        self.LeapId = p.loadURDF(
            path_src,
            # [-0.05, -0.03, -0.125],
            [0.0,0.038,0.098],
            p.getQuaternionFromEuler([0, -1.57 , 0]),
            useFixedBase = True
        )

        self.numJoints = p.getNumJoints(self.LeapId)
        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)
        self.create_target_vis()

        for i in range(self.numJoints):
            joint_info = p.getJointInfo(self.LeapId, i)
            link_name = joint_info[12].decode('utf-8')
            print(link_name)
            print(joint_info)

        self.operator2mano = OPERATOR2MANO_RIGHT 
        self.leap_node = LeapNode()
        self.oculus_reader = OculusReader()
        self.joint_names = [
            "Index1",
            "Index2",
            "Index3",
            "IndexTip",
            "Middle1",
            "Middle2",
            "Middle3",
            "MiddleTip",
            "Ring1",
            "Ring2",
            "Ring3",
            "RingTip",
            "Thumb1",
            "Thumb2",
            "Thumb3",
            "ThumbTip"
        ]

            
    def create_target_vis(self):
        # load balls
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]
        self.ballMbt = []
        for i in range(0,16):
            self.ballMbt.append(p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)) # for base and finger tip joints    
            no_collision_group = 0
            no_collision_mask = 0
            p.setCollisionFilterGroupMask(self.ballMbt[i], -1, no_collision_group, no_collision_mask)
            p.changeVisualShape(self.ballMbt[i], -1, rgbaColor=[1, 0, 0, 1]) 
        
    def update_target_vis(self, hand_pos):
        for i in range(len(hand_pos)):
            _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[i])
            p.resetBasePositionAndOrientation(self.ballMbt[i], hand_pos[i], current_orientation)
        
    def compute_IK(self, hand_pos, rot):
        p.stepSimulation()     

        index_mcp_pos = hand_pos[0]
        index_pip_pos = hand_pos[1]
        index_dip_pos = hand_pos[2]
        index_tip_pos = hand_pos[3]
        middle_mcp_pos = hand_pos[4]
        middle_pip_pos = hand_pos[5]
        middle_dip_pos = hand_pos[6]
        middle_tip_pos = hand_pos[7]
        ring_mcp_pos = hand_pos[8]
        ring_pip_pos = hand_pos[9]
        ring_dip_pos = hand_pos[10]
        ring_tip_pos = hand_pos[11]
        thumb_mcp_pos = hand_pos[12]
        thumb_pip_pos = hand_pos[13]
        thumb_dip_pos = hand_pos[14]
        thumb_tip_pos = hand_pos[15]

        index_mcp_rot = rot[0]
        index_pip_rot = rot[1]
        index_dip_rot = rot[2]
        index_tip_rot = rot[3]
        middle_mcp_rot = rot[4]
        middle_pip_rot = rot[5]
        middle_dip_rot = rot[6]
        middle_tip_rot = rot[7]
        ring_mcp_rot = rot[8]
        ring_pip_rot = rot[9]
        ring_dip_rot = rot[10]
        ring_tip_rot = rot[11]
        thumb_mcp_rot = rot[12]
        thumb_pip_rot = rot[13]
        thumb_dip_rot = rot[14]
        thumb_tip_rot = rot[15]
        
        # leapEndEffectorPos = [
        #     index_mcp_pos,
        #     index_pip_pos,
        #     index_dip_pos,
        #     index_tip_pos,
        #     middle_mcp_pos,
        #     middle_pip_pos,
        #     middle_dip_pos,
        #     middle_tip_pos,
        #     ring_mcp_pos,
        #     ring_pip_pos,
        #     ring_dip_pos,
        #     ring_tip_pos,
        #     thumb_mcp_pos,
        #     thumb_pip_pos,
        #     thumb_dip_pos,
        #     thumb_tip_pos
        # ]

        # leapEndEffectorRot = [
        #     index_mcp_rot,
        #     index_pip_rot,
        #     index_dip_rot,
        #     index_tip_rot,
        #     middle_mcp_rot,
        #     middle_pip_rot,
        #     middle_dip_rot,
        #     middle_tip_rot,
        #     ring_mcp_rot,
        #     ring_pip_rot,
        #     ring_dip_rot,
        #     ring_tip_rot,
        #     thumb_mcp_rot,
        #     thumb_pip_rot,
        #     thumb_dip_rot,
        #     thumb_tip_rot
        # ]

        leapEndEffectorPos = [
            index_tip_pos,
            middle_tip_pos,
            ring_tip_pos,
            thumb_tip_pos
        ]

        leapEndEffectorRot = [
            index_tip_rot,
            middle_tip_rot,
            ring_tip_rot,
            thumb_tip_rot
        ]

        jointPoses = p.calculateInverseKinematics2(
            self.LeapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            leapEndEffectorRot,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=0.0001,
        )
        
        combined_jointPoses = (jointPoses[0:4] + (0.0,) + jointPoses[4:8] + (0.0,) + jointPoses[8:12] + (0.0,) + jointPoses[12:16] + (0.0,))
        # combined_jointPoses = (jointPoses[0:16])
        combined_jointPoses = list(combined_jointPoses)

        # update the hand joints
        for i in range(20):
            p.setJointMotorControl2(
                bodyIndex=self.LeapId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=combined_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        real_robot_hand_q[0:4] = jointPoses[0:4]
        real_robot_hand_q[4:8] = jointPoses[4:8]
        real_robot_hand_q[8:12] = jointPoses[8:12]
        real_robot_hand_q[12:16] = jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        self.leap_node.set_allegro(real_robot_hand_q)

    def operation(self):
        while True:
            joint_pos = self.oculus_reader.get_joint_transformations()[1]
            pos = []
            final_pos = []
            rot = []
            if joint_pos == {}:
                print("None")
                continue
            else:
                # print("Receive info")
                mediapipe_wrist_rot = joint_pos["WristRoot"][:3, :3]
                wrist_position = joint_pos["WristRoot"][:3, 3]
                # wrist_position[2] += 0.05
                
                for joint_name in self.joint_names:
                    joint_transformation = joint_pos[joint_name]
                
                    pos.append(joint_transformation[:3, 3] - wrist_position)
                    if joint_name == "MiddleTip":
                        pos[-1][0] += 0.009375
                    if joint_name == "RingTip":
                        pos[-1][0] -= 0.004375
                    if joint_name == "IndexTip":
                        pos[-1][0] -= 0.00125
                    if joint_name == "ThumbTip":
                        pos[-1][0] += 0.00375
                    pos[-1] = pos[-1] @ mediapipe_wrist_rot @ self.operator2mano
                        
                    # Turn the rotation matrix into quaternion
                    rotation = joint_transformation[:3, :3] @ mediapipe_wrist_rot @ self.operator2mano
                    quaternion = R.from_matrix(rotation).as_quat()
                    rot.append(quaternion)

                    final_pos.append([pos[-1][0] * self.glove_to_leap_mapping_scale * 1.15, pos[-1][1] * self.glove_to_leap_mapping_scale, pos[-1][2] * self.glove_to_leap_mapping_scale])
                    final_pos[-1][2] -= 0.05
            # final_pos[3][2] += 0.005
            # final_pos[7][2] -= 0.005
            # final_pos[11][2] += 0.005
            # final_pos[15][2] -= 0.01
            
            self.compute_IK(final_pos, rot)     
            self.update_target_vis(final_pos)
            
        

def main(args=None):
    leappybulletik = LeapPybulletIK()
    leappybulletik.operation()

if __name__ == "__main__":
    main()