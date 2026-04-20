#!/usr/bin/env python3

import os
import time
import math
import threading
import numpy as np
import pybullet as p

import rospy
from oculus_reader.scripts import *
from oculus_reader.scripts.reader import OculusReader

from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils as lhu
from scipy.spatial.transform import Rotation as R

import actionlib
import kinova_msgs.msg
import std_msgs.msg
import geometry_msgs.msg

# import keyboard
from pynput.keyboard import Key, Listener
'''
这个程序获取手套数据，运行逆运动学并发布到LEAP手上。
机械臂控制被修改为使用pose_action_client而不是joints_action_client。

注意指尖位置是匹配的，但两只手之间的关节角度并不相同。:) 

灵感来自Wang等人的Dexcap https://dex-cap.github.io/ 和Shaw等人的Robotic Telekinesis。
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

camera_base_rot =np.array([
    [-1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
])

leap2human_rot = np.array([
    [-1, 0, 0],
    [0, 0, -1],
    [0, -1, 0]
]) 

correction_rot = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
])

reflection_rot = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])

Rx_180 = np.array([
    [1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1]
]) 

# 将3*3矩阵转换为四元数
def mat2quat(mat):
    q = np.zeros([4])
    q[3] = np.sqrt(1 + mat[0][0] + mat[1][1] + mat[2][2]) / 2
    q[0] = (mat[2][1] - mat[1][2]) / (4 * q[3])
    q[1] = (mat[0][2] - mat[2][0]) / (4 * q[3])
    q[2] = (mat[1][0] - mat[0][1]) / (4 * q[3])
    return q

# 四元数归一化
def QuaternionNorm(Q_raw):
    qx_temp,qy_temp,qz_temp,qw_temp = Q_raw[0:4]
    qnorm = math.sqrt(qx_temp*qx_temp + qy_temp*qy_temp + qz_temp*qz_temp + qw_temp*qw_temp)
    qx_ = qx_temp/qnorm
    qy_ = qy_temp/qnorm
    qz_ = qz_temp/qnorm
    qw_ = qw_temp/qnorm
    Q_normed_ = [qx_, qy_, qz_, qw_]
    return Q_normed_

# VR ==> MJ映射，当遥操作用户站在机器人前面时
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

# VR ==> MJ映射，当遥操作用户在机器人后面时
def vrbehind2mj(pose):
    pos = camera_base_rot @ pose[:3, 3]
    mat = reflection_rot @ (Rx_180 @ (correction_rot @ (leap2human_rot @ (camera_base_rot @ pose[:3, :3])) @ correction_rot)) @ reflection_rot

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

class KinovaNode:
    def __init__(self):
        rospy.init_node('teleoperation_node', anonymous=True)
        
        # 关节配置
        self.controlled_joints = [
            'j2n6s300_joint_1', 'j2n6s300_joint_2',
            'j2n6s300_joint_3', 'j2n6s300_joint_4',
            'j2n6s300_joint_5', 'j2n6s300_joint_6'
        ]
        
        # ROS通信
        self.joint_state_sub = rospy.Subscriber('/j2n6s300_driver/out/joint_state', JointState, self.joint_state_callback, queue_size=1)
        self.cartesian_command_sub = rospy.Subscriber('/j2n6s300_driver/out/cartesian_command', kinova_msgs.msg.KinovaPose, self.cartesian_command_callback, queue_size=1)
        
        # 状态变量
        self.current_q = None
        self.current_cartesian_command = [0.212322831154, -0.257197618484, 0.509646713734, 1.63771402836, 1.11316478252, 0.134094119072]
        self.prefix = 'j2n6s300_'
        
        rospy.sleep(3)
        print("Kinova node init success")

        self.rospy_thread = threading.Thread(target=rospy.spin)
        self.rospy_thread.start()

    def joint_state_callback(self, msg):
        """关节状态回调函数"""
        q_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
        self.current_qpos = q_dict
        return
    
    def cartesian_command_callback(self, feedback):
        """笛卡尔坐标回调函数"""
        currentCartesianCommand_str_list = str(feedback).split("\n")
        for index in range(0,len(currentCartesianCommand_str_list)):
            temp_str=currentCartesianCommand_str_list[index].split(": ")
            self.current_cartesian_command[index] = float(temp_str[1])
    
    def cartesian_pose_client(self, position, orientation):
        """发送笛卡尔坐标目标到动作服务器"""
        action_address = '/' + self.prefix + 'driver/pose_action/tool_pose'
        client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmPoseAction)
        client.wait_for_server()

        goal = kinova_msgs.msg.ArmPoseGoal()
        goal.pose.header = std_msgs.msg.Header(frame_id=(self.prefix + 'link_base'))
        goal.pose.pose.position = geometry_msgs.msg.Point(
            x=position[0], y=position[1], z=position[2])
        goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(
            x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])

        client.send_goal(goal)

        if client.wait_for_result(rospy.Duration(10.0)):
            return client.get_result()
        else:
            client.cancel_all_goals()
            print('        笛卡尔动作超时')
            return None

    def publish_cartesian_command(self, target_pos, target_quat):
        """发布笛卡尔坐标指令"""        
        rospy.loginfo(f"发布笛卡尔坐标指令 - 位置: {target_pos}, 姿态: {target_quat}")
        result = self.cartesian_pose_client(target_pos, target_quat)
        return result

class LeapNode:
    def __init__(self):
        ####一些参数
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
           
        #您可以在这里放置正确的端口，或让节点在前3个端口自动搜索手部设备
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB2', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
                self.dxl_client.connect()
        
        #启用位置-电流控制模式和默认参数，它命令一个位置然后限制电流，这样电机不会过载
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain刚度     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # 侧向刚度应该稍微小一点
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain阻尼
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # 侧向阻尼应该稍微小一点
        #最大电流（单位1ma），这样不会过热和抓得太紧 #500正常或#350轻型
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        
    #接收LEAP姿态并直接控制机器人
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        
    #allegro兼容性
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        
    #仿真兼容性，首先读取范围[-1,1]内的仿真值，然后转换为leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        
    #读取位置
    def read_pos(self):
        return self.dxl_client.read_pos()
    
    #读取速度
    def read_vel(self):
        return self.dxl_client.read_vel()
    
    #读取电流
    def read_cur(self):
        return self.dxl_client.read_cur()

class SystemPybulletIK():
    def __init__(self):
        # 启动pybullet
        p.connect(p.GUI)
        # 加载右侧leap手      
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        self.glove_to_leap_mapping_scale = 1.6
        self.leapEndEffectorIndex = [4,9,14,19]
        self.kinovaEndEffectorIndex = 9
        self.kinova_node = KinovaNode()
        kinova_path_src = "/media/yaxun/manipulation1/leaphandProject/kinova-ros/kinova_description/urdf/robot.urdf"
        self.kinovaId = p.loadURDF(
            kinova_path_src,
            [0.5, 0.0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase = True
        )

        leap_path_src = os.path.join(path_src, "leap_hand_mesh_right/robot_pybullet.urdf")
        self.leapId = p.loadURDF(
            leap_path_src,
            [0.0,0.038,0.098],
            p.getQuaternionFromEuler([0, -1.57 , 0]),
            useFixedBase = True
        )

        self.numJoints = p.getNumJoints(self.kinovaId)
        self.leapnumJoints = p.getNumJoints(self.leapId)
        for i in range(2,8):
            p.resetJointState(self.kinovaId, i, np.pi)

        for i in range(0, self.numJoints):
            print(p.getJointInfo(self.kinovaId, i))

        for i in range(0, self.leapnumJoints):
            print(p.getJointInfo(self.leapId, i))

        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)
        # self.create_target_vis()

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
        # 加载球体
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]
        self.ballMbt = []
        for i in range(0,16):
            self.ballMbt.append(p.createMultiBody(baseMass, ball_shape, -1, basePosition, [0,0,0,1]))
        
    def update_target_vis(self, hand_pos):
        for i in range(len(hand_pos)):
            p.resetBasePositionAndOrientation(self.ballMbt[i], hand_pos[i], [0,0,0,1])
        
    def compute_IK(self, hand_pos, rot, target_pos, target_quat):
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

        leap_jointPoses = p.calculateInverseKinematics2(
            self.leapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            leapEndEffectorRot,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=0.0001,
        )

        # 更新机械手关节
        for i in range(0,6):
            p.resetJointState(self.kinovaId, i+2, 0)
        
        for i in range(10,30):
            p.resetJointState(self.leapId, i, leap_jointPoses[i-10])
        
        for i in range(self.leapnumJoints):
            p.resetJointState(self.leapId, i, leap_jointPoses[i])

        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        real_robot_hand_q[0:4] = leap_jointPoses[0:4]
        real_robot_hand_q[4:8] = leap_jointPoses[4:8]
        real_robot_hand_q[8:12] = leap_jointPoses[8:12]
        real_robot_hand_q[12:16] = leap_jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        self.leap_node.set_allegro(real_robot_hand_q)

        print("目标位置:", target_pos)
        print("目标四元数:", target_quat)
        
        # 使用pose action client控制机械臂
        result = self.kinova_node.publish_cartesian_command(target_pos, target_quat)
        if result:
            print("笛卡尔姿态指令发送成功!")
        else:
            print("笛卡尔姿态指令发送失败")

    def operation(self):
        VRP0 = None
        VRR0 = None
        MJP0 = None
        MJR0 = None
        while True:
            try:
                transforms = self.oculus_reader.get_transformations_and_buttons()
                r_controller_T44 = transforms['r_controller_T44']
                left_controller_T44 = transforms['l_controller_T44']
                r_pose_wrist, r_rot_wrist = vrbehind2mj(r_controller_T44)
                l_pose_wrist, l_rot_wrist = vrbehind2mj(left_controller_T44)
                button_states = transforms['buttons']
                if button_states['RIndexTrigger'][0] > 0.5 and space_pressed:
                    target_pos = r_pose_wrist
                    target_quat = r_rot_wrist
                    hand_pos, rot = self.oculus_reader.get_joint_poses()
                    self.compute_IK(hand_pos, rot, target_pos, target_quat)
            except Exception as e:
                print(f"操作中出现错误: {e}")
                pass

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
    
    leappybulletik = SystemPybulletIK()
    leappybulletik.operation()

if __name__ == "__main__":
    main()