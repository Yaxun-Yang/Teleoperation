import numpy as np
import socket
import numpy as np
import pickle
import pybullet as p
import operator
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time
import math

import multiprocessing
import time
from pathlib import Path
from queue import Empty
from pynput.keyboard import Key, Listener

import cv2
import numpy as np

from loguru import logger

from mediapipe_hand_detector import SingleHandDetector

import pyk4a
from pyk4a import Config, PyK4A

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
#######################################################
"""This can control and query the LEAP Hand

I recommend you only query when necessary and below 90 samples a second.  Used the combined commands if you can to save time.  Also don't forget about the USB latency settings in the readme.

#Allegro hand conventions:
#0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more.

#LEAP hand conventions:
#180 is flat out home pose for the index, middle, ring, finger MCPs.
#Applying a positive angle closes the joints more and more to curl closed.
#The MCP is centered at 180 and can move positive or negative to that.

#The joint numbering goes from Index (0-3), Middle(4-7), Ring(8-11) to Thumb(12-15) and from MCP Side, MCP Forward, PIP, DIP for each finger.
#For instance, the MCP Side of Index is ID 0, the MCP Forward of Ring is 9, the DIP of Ring is 11

"""
HOST = '172.16.0.201' 
#HOST =  '127.0.0.1' # 本地回环地址
PORT = 65433        # 监听端口
camera_matrix = np.loadtxt('cam_K.txt').reshape(3,3)
space_pressed = False

#create a  q_offet [-0.02707737  0.81567121 -0.37449085  0.44011805]
#This is the q_offset for the leap hand to match the kinova hand
q_offset = np.array([-0.02707737, 0.81567121, -0.37449085, 0.44011805])
# q_offset = np.array([0.53468345, -0.49341045, -0.34625187, 0.59225787])

class LeapNode:
    def __init__(self):
        ####Some parameters
        # I recommend you keep the current limit from 350 for the lite, and 550 for the full hand
        # Increase KP if the hand is too weak, decrease if it's jittery.
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        # For example ls /dev/serial/by-id/* to find your LEAP Hand. Then use the result.  
        # For example: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7W91VW-if00-port0
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            # self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client = DynamixelClient(motors, '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U1QV-if00-port0', 4000000)
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
    #allegro compatibility joint angles.  It adds 180 to make the fully open position at 0 instead of 180
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position of the robot
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()
    #These combined commands are faster FYI and return a list of data
    def pos_vel(self):
        return self.dxl_client.read_pos_vel()
    #These combined commands are faster FYI and return a list of data
    def pos_vel_eff_srv(self):
        return self.dxl_client.read_pos_vel_cur()


def quaternion_inverse(q):
    """计算单位四元数的逆（共轭）"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    """四元数乘法（顺序：q1 ⊗ q2）"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def compute_rotation_offset(q_current, q_target):
    """计算从当前旋转到目标旋转的四元数偏移量"""
    q_current_inv = quaternion_inverse(q_current)
    q_offset = quaternion_multiply(q_target, q_current_inv)
    # 可选：归一化以消除计算误差
    q_offset /= np.linalg.norm(q_offset)
    return q_offset

class KinovaPybulletIK():
    def __init__(self):
        # start pybullet
        #clid = p.connect(p.SHARED_MEMORY)
        #clid = p.connect(p.DIRECT)
        p.connect(p.GUI)
        # load right leap hand      
        # path_src = os.path.abspath(__file__)
        # path_src = os.path.dirname(path_src)
        self.glove_to_leap_mapping_scale = 1.6
        self.kinovaEndEffectorIndex = 9

        # path_src = os.path.join(path_src, "leap_hand_mesh_right/robot_pybullet.urdf")
        path_src = "/media/yaxun/manipulation1/leaphandProject/kinova-ros/kinova_description/urdf/robot.urdf"
        leap_path_src = "leap_hand_mesh_right/robot_pybullet.urdf"
        self.leapEndEffectorIndex = [4,9,14,19]
        ##You may have to set this path for your setup on ROS2
        self.kinovaId = p.loadURDF(
            path_src,
            # [-0.05, -0.03, -0.125],
            # [0.0,0.038,0.098],
            [0.5, 0.0, 0.0],
            p.getQuaternionFromEuler([0, 0 , 0]),
            useFixedBase = True
        )
        self.LeapId = p.loadURDF(
            leap_path_src,
            # [-0.05, -0.03, -0.125],
            [0.0,0.0,0.0],
            p.getQuaternionFromEuler([0, -1.57, 0]),
            useFixedBase = True
        )
        self.numJoints = p.getNumJoints(self.kinovaId)
        self.leap_numJoints = p.getNumJoints(self.LeapId)
        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)
        self.create_target_vis()
        self.create_leap_vis()

        for i in range(2,8):
            p.resetJointState(self.kinovaId, i, np.pi)
        

        # p.resetJointState(self.kinovaId, 2, np.pi/2)
        # p.resetJointState(self.kinovaId, 3, np.pi/2+np.pi/4)
        # p.resetJointState(self.kinovaId, 4, np.pi+np.pi/4)
        # p.resetJointState(self.kinovaId, 5, 0)
        p.resetJointState(self.kinovaId, 6, 0)
        p.resetJointState(self.kinovaId, 7, 0)

        for i in range(self.numJoints):
            info = p.getJointInfo(self.kinovaId, i)
            print(info)

        for i in range(self.leap_numJoints):
            leap_joint_info = p.getJointInfo(self.LeapId, i)
            leap_link_name = leap_joint_info[12].decode('utf-8')
            print(leap_link_name)
            print(leap_joint_info)

        # self.operator2mano = OPERATOR2MANO_RIGHT 
        # self.leap_node = LeapNode()

        self.leap_joint_names = [
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
    
    def create_leap_vis(self):
        # load balls
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]
        self.ballMbt = []
        for i in range(0,21):
            self.ballMbt.append(p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)) # for base and finger tip joints    
            no_collision_group = 0
            no_collision_mask = 0
            p.setCollisionFilterGroupMask(self.ballMbt[i], -1, no_collision_group, no_collision_mask)
            p.changeVisualShape(self.ballMbt[i], -1, rgbaColor=[1, 0, 0, 1]) 
        
    def update_leap_vis(self, hand_pos):
        for i in range(len(hand_pos)):
            _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[i])
            p.resetBasePositionAndOrientation(self.ballMbt[i], hand_pos[i], current_orientation)

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
        return jointPoses


            

    def retartet_arm(self, move_vector, eef_q):
        current_end_effector_pos = p.getLinkState(bodyUniqueId = self.kinovaId,linkIndex = self.kinovaEndEffectorIndex)[4]
        # current_end_effector_q = p.getLinkState(bodyUniqueId = self.kinovaId,linkIndex = self.kinovaEndEffectorIndex)[5]
        # q_offset = compute_rotation_offset(wrist_q, current_end_effector_q)
        # print(f'q_offset: {q_offset}')
        # result = quaternion_multiply(q_offset, wrist_q)
       

        target_pos = current_end_effector_pos + move_vector
        # print(f'current_end_effector_pos: {current_end_effector_pos}')
        qpos = self.compute_IK(target_pos, eef_q)
        # print(qposs)
        # print(f'qpos: ')
        self.update_target_vis(target_pos)

        return qpos
    
    def mediapip_handpose_to_sim(self, pos):
        sim_pos = pos.copy()
        sim_pos[0] = pos[2]
        sim_pos[1] = pos[1]
        sim_pos[2] = -pos[0]
        return sim_pos

    def retarget_hand(self, hand_pos):
        p.stepSimulation() 
        current_end_effector_pos = p.getLinkState(bodyUniqueId = self.kinovaId,linkIndex = self.kinovaEndEffectorIndex)[4]
        # print(f'hand_pos: {hand_pos}')    
        # for i in range(len(hand_pos)):
        #     # hand_pos[i] = self.mediapip_handpose_to_sim(hand_pos[i])
        #     hand_pos[i][1] = hand_pos[i][1] - 0.03
        #     hand_pos[i][2] = hand_pos[i][2] - 0.08
        hand_pos = hand_pos *2
        hand_pos = hand_pos + np.array([-0.03, -0.04, -0.2])
        # print(f'hand_pos: {hand_pos}')

        self.update_leap_vis(hand_pos)
        index_tip_pos = hand_pos[8]
        middle_tip_pos = hand_pos[12]
        ring_tip_pos = hand_pos[16]
        thumb_tip_pos = hand_pos[4]
 
        leapEndEffectorPos = [
            index_tip_pos,
            middle_tip_pos,
            ring_tip_pos,
            thumb_tip_pos
        ]

        jointPoses = p.calculateInverseKinematics2(
            self.LeapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
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

        # real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        # real_robot_hand_q[0:4] = jointPoses[0:4]
        # real_robot_hand_q[4:8] = jointPoses[4:8]
        # real_robot_hand_q[8:12] = jointPoses[8:12]
        # real_robot_hand_q[12:16] = jointPoses[12:16]
        # real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        # real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        # real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        # self.leap_node.set_allegro(real_robot_hand_q)
        # return combined_jointPoses

    # def updata_sim(self, leap_qpos , arm_qpos):



def project_to_3d(uv_points, depth_map):
        """使用深度图将2D点转换为3D坐标"""
        points3d = []
        for u, v in uv_points:
            v = int(np.round(v))
            u = int(np.round(u))
            if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                z = depth_map[v, u]
                x = (u - camera_matrix[0,2]) * z / camera_matrix[0,0]
                y = (v - camera_matrix[1,2]) * z / camera_matrix[1,1]
                points3d.append([x, y, z])
            else:
                points3d.append([0, 0, 0])  # 无效点占位
        return np.array(points3d)

def kinect_to_kinova(kinect_pos):
    # Kinect坐标系到Kinova坐标系的转换矩阵
    kinova_pos = kinect_pos.copy()
    kinova_pos[0] = kinect_pos[0]
    kinova_pos[1] = kinect_pos[2]
    kinova_pos[2] = -kinect_pos[1]
    return kinova_pos

import numpy as np
import operator

def r_quaternion(R):
    # This function takes as input a 3x3 rotation matrix and returns the
    # corresponding unit quaternion in the format [x, y, z, w].
    # Implements Cayley's method based on the referenced paper.

    # Calculate the four components
    e0 = 0.25 * np.sqrt((1 + R[0,0] + R[1,1] + R[2,2])**2 + (R[2,1] - R[1,2])**2 + (R[0,2] - R[2,0])**2 + (R[1,0] - R[0,1])**2)
    e1 = 0.25 * np.sqrt((R[2,1] - R[1,2])**2 + (1 + R[0,0] - R[1,1] - R[2,2])**2 + (R[0,1] + R[1,0])**2 + (R[2,0] + R[0,2])**2)
    e2 = 0.25 * np.sqrt((R[0,2] - R[2,0])**2 + (R[0,1] + R[1,0])**2 + (1 - R[0,0] + R[1,1] - R[2,2])**2 + (R[1,2] + R[2,1])**2)
    e3 = 0.25 * np.sqrt((R[1,0] - R[0,1])**2 + (R[2,0] + R[0,2])**2 + (R[1,2] + R[2,1])**2 + (1 - R[0,0] - R[1,1] + R[2,2])**2)
    
    e = [e0, e1, e2, e3]
    index, max_value = max(enumerate(e), key=operator.itemgetter(1))
    
    # Adjust signs based on the largest component
    if index == 0:
        e[0] = e0
        e[1] = np.sign(R[2,1] - R[1,2]) * e1
        e[2] = np.sign(R[0,2] - R[2,0]) * e2
        e[3] = np.sign(R[1,0] - R[0,1]) * e3
    elif index == 1:
        e[0] = np.sign(R[2,1] - R[1,2]) * e0
        e[1] = e1
        e[2] = np.sign(R[0,1] + R[1,0]) * e2
        e[3] = np.sign(R[0,2] + R[2,0]) * e3
    elif index == 2:
        e[0] = np.sign(R[0,2] - R[2,0]) * e0
        e[1] = np.sign(R[0,1] + R[1,0]) * e1
        e[2] = e2
        e[3] = np.sign(R[1,2] + R[2,1]) * e3
    else:
        e[0] = np.sign(R[1,0] - R[0,1]) * e0
        e[1] = np.sign(R[0,2] + R[2,0]) * e1
        e[2] = np.sign(R[1,2] + R[2,1]) * e2
        e[3] = e3
    
    # Reorder from [w, x, y, z] to [x, y, z, w]
    quat = [e[1], e[2], e[3], e[0]]
    
    # Normalize and return
    return np.array(quat) / np.linalg.norm(quat)

def rotation_matrix_to_euler_angles_xyz(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (XYZ convention - roll, pitch, yaw)
    
    Parameters:
    R : numpy.ndarray
        3x3 rotation matrix
        
    Returns:
    numpy.ndarray
        Euler angles in radians [roll, pitch, yaw] (X, Y, Z rotations)
    """
    # Calculate pitch (Y rotation)
    sy = math.sqrt(R[0,0] * R[0,0] + R[0,1] * R[0,1])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = math.atan2(R[2,1], R[2,2])  # X rotation
        pitch = math.atan2(-R[2,0], sy)    # Y rotation
        yaw = math.atan2(R[1,0], R[0,0])   # Z rotation
    else:
        roll = math.atan2(-R[1,2], R[1,1])  # X rotation
        pitch = math.atan2(-R[2,0], sy)     # Y rotation
        yaw = 0                             # Z rotation

    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    
    return np.array([roll, pitch, yaw])



def start_retargeting(queue: multiprocessing.Queue, robot_dir: str):
    print(f'root dir: {robot_dir}')
    global space_pressed
    # leap_hand = LeapNode()
    kinovapybulletik = KinovaPybulletIK()
    
    hand_type = "Right"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    init_stage = True
    least_wrist_pos = [0,0,0]
    K = np.loadtxt('cam_K.txt').reshape(3,3)
    # k4a init
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            camera_fps=pyk4a.FPS.FPS_30,
            synchronized_images_only=True,
        )
    )
    k4a.start()
    calibration = k4a.calibration

    K = calibration.get_camera_matrix(1) # stand for color type
    window_name = 'k4a'
    start_trigger = False
    annotation = False
    first_tracking_frame = False
    index = 0
    zfar = 2.0
    first_downscale = True
    shorter_side = 720
    recording = True
    first_recording = True

    # Different robot loader may have different orders for joints
    # retargeting_joint_names = retargeting.joint_names
    # retargeting_to_urdf = np.array([retargeting_joint_names.index(name) for name in ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]]).astype(int)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))     # 绑定地址和端口
        s.listen()               # 开始监听
        print(f"Server正在监听 {HOST}:{PORT}...")
        # conn, addr = s.accept()
        conn , addr = None, None
        if conn is None:
            print('已连接客户端:', addr)
            # 使用pickle序列化numpy数组并发送
            while True:
                try:
                    # data = queue.get(timeout=5)  # 从队列获取数据
                    # bgr = data['image']
                    # depth = data['depth']
                    capture = k4a.get_capture()

                    if first_downscale:
                        H, W = capture.color.shape[:2]
                        downscale = shorter_side / min(H, W)
                        H = int(H*downscale)
                        W = int(W*downscale)
                        K[:2] *= downscale
                        first_downscale = False        
                
                
                    color = capture.color[...,:3].astype(np.uint8)
                    bgr = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST) 
                    depth = capture.transformed_depth.astype(np.float32) / 1e3
                    depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
                    depth[(depth<0.01) | (depth>=zfar)] = 0
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                except Empty:
                    logger.error(f"Fail to fetch image from camera in 5 secs. Please check your web camera device.")
                    return

                _, joint_pos, keypoint_2d, wrist_rot ,keypoint_3d= detector.detect(rgb)
                w_keypoint_3d = keypoint_3d
                bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
                cv2.imshow("realtime_retargeting_demo", bgr)
                if cv2.waitKey(1) & 0xFF == ord("s"):
                    break

                if joint_pos is None:
                    logger.warning(f"{hand_type} hand is not detected.")
                else :
                    indices = np.array([[0,0,0,0,0],[0,4,8,12,16]])
                    # print(f'indices: {indices}')
                    # print(f'indexes shape: {indices.shape}')
                    
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    # make human hand more like leap hand. adjust wrist pos hand finget tip length
                    joint_pos[12] = joint_pos[11]+(joint_pos[12]-joint_pos[11])*0.45
                    joint_pos[16] = joint_pos[15]+(joint_pos[15]-joint_pos[14])*0.95
                    # print(joint_pos[task_indices, :])
                    # print(f'joint 2 ------------: {joint_pos[2]}')
                    # print(f'joint 9 ------------: {joint_pos[9]}')
                    # # make wrist pos is 0,0,0.05
                    # exit()
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    ref_value = ref_value * 1

                    # qpos = retargeting.retarget(ref_value)

                    keypoint_2d_array = detector.parse_keypoint_2d(keypoint_2d,bgr.shape[:2])
                    uvpoints = []
                    # only have wrist 
                    uvpoints.append(keypoint_2d_array[0])
                    points3d = project_to_3d(uvpoints,depth)
                    if init_stage : 
                        least_wrist_pos = points3d[0]
                        move_vector = np.array([0,0,0])
                        init_stage = False
                    else : 
                        move_vector = kinect_to_kinova(points3d[0]) - kinect_to_kinova(least_wrist_pos)
                        move_vector_length = np.linalg.norm(move_vector)
                        print(f'move_vector: {move_vector}')
                        print(f'move_vector length: {move_vector_length}')
                        # wrist_rot [3,3] to quaternion 
                        # print(f'wrist_rot: {wrist_rot}')
                        print(f'wrist_rot (euler): {rotation_matrix_to_euler_angles_xyz(wrist_rot)}')
                    
                        # wrist_rot = C_T @ wrist_rot @ C
                        # wrist_rot =wrist_rot.T
                        # wrist_rot = np.array([
                        #     [0, -1, 0],
                        #     [1, 0, 0],
                        #     [0, 0, 1]
                        # ])
                        Ry = np.array([
                            [ 0,  0,  1],
                            [ 0,  1,  0],
                            [-1,  0,  0]
                        ])
                        R_y = np.array([
                            [ 0,  0,  -1],
                            [ 0,  1,  0],
                            [1,  0,  0]
                        ])
                        Rz = np.array([
                        [0,  -1,  0],
                        [1, 0,  0],
                        [ 0,  0,  1]
                        ])  
                        R_z = np.array([
                        [0,  1,  0],
                        [-1, 0,  0],
                        [ 0,  0,  1]
                        ]) 
                        Rz_180 = np.array([
                        [-1,  0,  0],
                        [0, -1,  0],
                        [ 0,  0,  1]
                        ])
                        Rx = np.array([
                        [1,  0,  0],
                        [ 0, 0,  -1],
                        [ 0,  1,  0]])  
                        R_x = np.array([
                        [1,  0,  0],
                        [ 0, 0,  1],
                        [ 0,  -1,  0]])  
                        Rx_180 = np.array([
                        [1,  0,  0],
                        [ 0, -1,  0],
                        [ 0,  0, -1]]) 
                        leap_eef_tf = np.array([
                            [0, 0, -1],
                            [0, -1, 0],
                            [-1, 0, 0]
                        ])
                        test_yannick = np.array([
                            [0, 0, -1],
                            [0, -1, 0],
                            [-1, 0, 0]
                        ])
                        # w_wrist_rot = Rz @ R_y @ Rz @ R_y @ wrist_rot
                        w_wrist_rot =  R_x @ wrist_rot
                        # eef_rot =  w_wrist_rot 
                        eef_rot = np.array([
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]
                        ])
                        # eef_rot =  Ry @ Rx_180 
                        eef_rot = w_wrist_rot @ R_y @ eef_rot
                        eef_q= r_quaternion(eef_rot)
                        
                        # print(f'eef_q {eef_q}')

                        w_keypoint_3d =(R_x @ keypoint_3d.T).T
                        w_keypoint_3d =(w_wrist_rot.T @ w_keypoint_3d.T).T
                        # w_keypoint_3d =( Ry @ w_keypoint_3d.T).T
                        # w_keypoint_3d =keypoint_3d @ Rx
                        # wrist_q = np.array([1,0,0,0])
                        # print(f'wrist_rot: {wrist_q}')
                        # wrist_q = quaternion_multiply(-1*q_offset, wrist_q)
                        # wrist_q[2] = -wrist_q[2]
                        # wrist_q[3] = -wrist_q[3]
                        # wrist_q[1] = -wrist_q[1]


                        if move_vector_length <0.006:
                            np.array([0,0,0])
                        print(f'space_pressed: {space_pressed}')
                        if True:
                            print("start control !!!")
                            arm_qpos = kinovapybulletik.retartet_arm(10*move_vector, eef_q)
                        least_wrist_pos = points3d[0]
                    # print(f'qpos: {qpos}')
                    leap_qpos = kinovapybulletik.retarget_hand(w_keypoint_3d)
                    
                    data_to_send = {
                        "move_vector": move_vector,
                        "wrist_rot": wrist_rot
                    }
                    
                    # serialized_data = pickle.dumps(data_to_send)
                    # conn.sendall(serialized_data)
                    # print("已发送数据:", data_to_send)

                    # leap_hand.set_allegro(qpos[retargeting_to_urdf])

                    time.sleep(0.03)




def produce_frame(queue: multiprocessing.Queue):

    K = np.loadtxt('cam_K.txt').reshape(3,3)
    # k4a init
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            camera_fps=pyk4a.FPS.FPS_30,
            synchronized_images_only=True,
        )
    )
    k4a.start()
    calibration = k4a.calibration

    K = calibration.get_camera_matrix(1) # stand for color type
    window_name = 'k4a'
    start_trigger = False
    annotation = False
    first_tracking_frame = False
    index = 0
    zfar = 2.0
    first_downscale = True
    shorter_side = 720
    recording = True
    first_recording = True

    while True :
        capture = k4a.get_capture()

        if first_downscale:
            H, W = capture.color.shape[:2]
            downscale = shorter_side / min(H, W)
            H = int(H*downscale)
            W = int(W*downscale)
            K[:2] *= downscale
            first_downscale = False        
     
     
        color = capture.color[...,:3].astype(np.uint8)
        color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST) 
        depth = capture.transformed_depth.astype(np.float32) / 1e3
        depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.01) | (depth>=zfar)] = 0

        queue.put({
            'image': color,
            'depth': depth
        })
        time.sleep(1 / 30.0)

def on_press(key):
    global space_pressed
    if key == Key.space:
        print("on_press")
        space_pressed = True
        print(f'space_pressed: {space_pressed}')

# 键盘释放回调
def on_release(key):
    global space_pressed
    if key == Key.space:
        space_pressed = False    


def main():
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    # print(f'hand type: {hand_type}')
    # print(hand_type.name)
    # exit()
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "dex-urdf" / "robots" / "hands" 



    queue = multiprocessing.Queue(maxsize=1000)
    # producer_process = multiprocessing.Process(target=produce_frame, args=(queue,))
    consumer_process = multiprocessing.Process(target=start_retargeting, args=(queue, str(robot_dir)))
    keyboard_listener = Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()

    # producer_process.start()
    consumer_process.start()

    # producer_process.join()
    consumer_process.join()
    keyboard_listener.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    main()
