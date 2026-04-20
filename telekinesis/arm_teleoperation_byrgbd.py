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

# import keyboard
from pynput.keyboard import Key, Listener
import rospy
'''
This takes the glove data, and runs inverse kinematics and then publishes onto LEAP Hand.

Note how the fingertip positions are matching, but the joint angles between the two hands are not.  :) 

Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
'''

space_pressed = False

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

# Convert 3*3 matrix to quaternion
def mat2quat(mat):
    q = np.zeros([4])
    q[0] = np.sqrt(1 + mat[0][0] + mat[1][1] + mat[2][2]) / 2
    q[1] = (mat[2][1] - mat[1][2]) / (4 * q[0])
    q[2] = (mat[0][2] - mat[2][0]) / (4 * q[0])
    q[3] = (mat[1][0] - mat[0][1]) / (4 * q[0])
    return q

# VR ==> MJ mapping when teleOp user is standing infront of the robot
def vrfront2mj(pose):
    pos = np.zeros([3])
    pos[0] = -1.*pose[2][3]
    pos[1] = -1.*pose[0][3]
    pos[2] = +1.*pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = -1.*pose[2][:3]
    mat[1][:] = +1.*pose[0][:3]
    mat[2][:] = -1.*pose[1][:3]

    return pos, mat2quat(mat)

# VR ==> MJ mapping when teleOp user is behind the robot
def vrbehind2mj(pose):
    pos = np.zeros([3])
    pos[0] = +1.*pose[2][3]
    pos[1] = +1.*pose[0][3]
    pos[2] = +1.*pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = +1.*pose[2][:3]
    mat[1][:] = -1.*pose[0][:3]
    mat[2][:] = -1.*pose[1][:3]

    return pos, mat2quat(mat)

def vr2mj(pose):
    pos = np.zeros([3])
    pos[0] = pose[0][3]
    pos[1] = pose[1][3]
    pos[2] = pose[2][3]
    pos = pos @ OPERATOR2MANO_RIGHT
    mat = pose[:3,:3] @ OPERATOR2MANO_RIGHT
    return pos, mat2quat(mat)

def negQuat(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])

def mulQuat(qa, qb):
    res = np.zeros(4)
    res[0] = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3]
    res[1] = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2]
    res[2] = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1]
    res[3] = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0]
    return res

def diffQuat(quat1, quat2):
    neg = negQuat(quat1)
    diff = mulQuat(quat2, neg)
    return diff

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

class KinovaPybulletIK():
    def __init__(self):
        # start pybullet
        #clid = p.connect(p.SHARED_MEMORY)
        #clid = p.connect(p.DIRECT)
        p.connect(p.GUI)
        # load right leap hand      
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        self.glove_to_leap_mapping_scale = 1.6
        self.kinovaEndEffectorIndex = 9

        # path_src = os.path.join(path_src, "leap_hand_mesh_right/robot_pybullet.urdf")
        path_src = "/media/yaxun/manipulation1/leaphandProject/kinova-ros/kinova_description/urdf/robot.urdf"
        ##You may have to set this path for your setup on ROS2
        self.kinovaId = p.loadURDF(
            path_src,
            # [-0.05, -0.03, -0.125],
            # [0.0,0.038,0.098],
            [0.0, 0.0, 0.0],
            p.getQuaternionFromEuler([0, 0 , 0]),
            useFixedBase = True
        )

        self.numJoints = p.getNumJoints(self.kinovaId)
        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)
        self.create_target_vis()

        for i in range(2,8):
            p.resetJointState(self.kinovaId, i, np.pi)

        for i in range(self.numJoints):
            info = p.getJointInfo(self.kinovaId, i)
            print(info)

        self.operator2mano = OPERATOR2MANO_RIGHT 
        # self.leap_node = LeapNode()
        self.oculus_reader = OculusReader()

        
    def update_target_vis(self, hand_pos):
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[0])
        p.resetBasePositionAndOrientation(self.ballMbt[0], hand_pos, current_orientation)
        
    def create_target_vis(self):
        # load balls
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]
        
        self.ballMbt = []
        for i in range(0,1):
            self.ballMbt.append(p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)) # for base and finger tip joints    
            no_collision_group = 0
            no_collision_mask = 0
            p.setCollisionFilterGroupMask(self.ballMbt[i], -1, no_collision_group, no_collision_mask)
        p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1]) 
    
    def compute_IK(self, arm_pos, arm_rot):
        p.stepSimulation()     

        jointPoses = p.calculateInverseKinematics(
            self.kinovaId,
            self.kinovaEndEffectorIndex,
            arm_pos,
            arm_rot,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=0.0001,
        )

        # update the hand joints
        for i in range(6):
            p.setJointMotorControl2(
                bodyIndex=self.kinovaId,
                jointIndex=i+2,
                controlMode=p.POSITION_CONTROL,
                targetPosition=jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

    def operation(self):
        VRP0 = None
        VRR0 = None
        MJP0 = None
        MJR0 = None
        while True:
            joint_pos = self.oculus_reader.get_joint_transformations()[1]
            transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
            # kinova
            if transformations and 'r' in transformations:     
                right_controller_pose = transformations['r']
                VRpos, VRquat = vrfront2mj(right_controller_pose) # front x, left y, up z
                # VRpos, VRquat = vrbehind2mj(right_controller_pose) # front -x, left -y, up z
                # VRpos, VRquat = vr2mj(right_controller_pose)
                if space_pressed:
                    # dVRP/R = VRP/Rt - VRP/R0
                    dVRP = (VRpos - VRP0)          
                    # dVRR = VRquat - VRR0
                    dVRR = diffQuat(VRR0, VRquat)
                    # MJP/Rt =  MJP/R0 + dVRP/R

                    curr_pos = MJP0 + dVRP
                    curr_quat = mulQuat(MJR0, dVRR)

                # Adjust origin if not engaged
                else:
                    # RP/R0 = RP/Rt
                    link_state = p.getLinkState(self.kinovaId, 9)
                    print("link_state", link_state)
                    curr_pos = link_state[4]
                    curr_quat = link_state[5]

                    for i in range(2,8):
                        print(p.getJointState(self.kinovaId, i)[0])

                    print("target_pos", curr_pos)
                    print("target_quat", curr_quat)
                    print("-----------------------------------")

                    MJP0 = curr_pos # real kinova pos origin
                    MJR0 = curr_quat # real kinova quat origin

                    # VP/R0 = VP/Rt
                    VRP0 = VRpos
                    VRR0 = VRquat

                # udpate desired pos
                target_pos = curr_pos
                target_quat =  curr_quat
            
                self.compute_IK(target_pos, target_quat)     
                self.update_target_vis(target_pos)
            

def on_press(key):
    global space_pressed
    if key == Key.space:
        space_pressed = True

# 键盘释放回调
def on_release(key):
    global space_pressed
    if key == Key.space:
        space_pressed = False     

def main(args=None):
    keyboard_listener = Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()
    
    kinovapybulletik = KinovaPybulletIK()
    kinovapybulletik.operation()

if __name__ == "__main__":
    main()