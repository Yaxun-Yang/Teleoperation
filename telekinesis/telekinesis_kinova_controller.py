#!/usr/bin/env python
import math
import rospy
import torch
import pytorch_kinematics as pk
import numpy as np
from sensor_msgs.msg import JointState
from kinova_msgs.msg import JointTorque
import actionlib
import kinova_msgs.msg
import time 

class KinovaIKController:
    def __init__(self, device='cuda'):
        rospy.init_node('kinova_ik_controller')
        
        # 硬件配置
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        
        # 运动链初始化
        urdf_path = './j2s6s300.urdf'
        self.chain = self._build_kinematic_chain(urdf_path)
        
        # 关节配置
        self.controlled_joints = [
            'j2n6s300_joint_1', 'j2n6s300_joint_2',
            'j2n6s300_joint_3', 'j2n6s300_joint_4',
            'j2n6s300_joint_5', 'j2n6s300_joint_6'
        ]
        
        # ROS通信
        #self.joint_command_pub = rospy.Publisher('/j2n6s300_driver//joints_action/joint_angles', JointTorque, queue_size=10)
        self.joint_state_sub = rospy.Subscriber('/j2n6s300_driver/out/joint_state', JointState, self.joint_state_callback)

        action_address = '/j2n6s300_driver/joints_action/joint_angles'
        self.client = actionlib.SimpleActionClient(action_address,
                                          kinova_msgs.msg.ArmJointAnglesAction)
        self.client.wait_for_server()
        
        
        # 状态变量
        self.current_q = None
        self.ik_solver = None
        self._init_ik_solver()
        # result = self._publish_joint_command(torch.tensor([math.pi/2, 3.1415927,3.1415927,0, 0, 0]).float().cuda())
        # print(result)
        # wait 5 sec
        rospy.sleep(3)
        print("init success")
        self._execute_movement(0.05)
        #init robot arm joint qpos
        #get current qpos , set goal qpos [0, 3.1415927,3.1415927,0, 0, 0]

    def _build_kinematic_chain(self, urdf_path):
        """构建运动学链并配置GPU支持"""
        chain = pk.build_serial_chain_from_urdf(
            open(urdf_path).read(), 
            end_link_name='j2n6s300_link_6',
            root_link_name='j2n6s300_link_base'
        )
        return chain.to(dtype=self.dtype, device=self.device)

    def _init_ik_solver(self):
        """初始化逆运动学求解器"""
        # joint_limits = torch.tensor(
        #     [[-3.14, 3.14]] * 6,  # 根据实际URDF修改限制
        #     dtype=self.dtype,
        #     device=self.device
        # )
        # get robot joint limits
        joint_limits = torch.tensor(self.chain.get_joint_limits(), device=self.device)

        self.ik_solver = pk.PseudoInverseIK(
            self.chain,
            max_iterations=100,
            num_retries=8,
            joint_limits=joint_limits.T,
            # position_tolerance=1e-4,
            # orientation_tolerance=1e-3,
            lr=0.02,
            # damping=1e-6
        )

    def joint_state_callback(self, msg):
        """关节状态回调函数"""
        
        self._process_initial_state(msg)
        # self._execute_movement(0.01)

    def _process_initial_state(self, msg):
        """处理初始关节状态"""
        q_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
        self.current_q = torch.tensor(
            [q_dict[name] for name in self.controlled_joints],
            dtype=self.dtype,
            device=self.device
        )
        with torch.no_grad():
            self.current_eef = self.chain.forward_kinematics(self.current_q.unsqueeze(0))
        #rospy.loginfo(f"current joint positions: {self.current_q.cpu().numpy()}")

    def _execute_movement(self, delta_z):
        """执行末端运动"""
        # 计算当前末端位姿
        print(f'current_q : {self.current_q}')

        start_time = time.time()
        # with torch.no_grad():
        #     current_tf = self.chain.forward_kinematics(self.current_q.unsqueeze(0))
        current_tf = self.current_eef.clone()
        
        # 构造目标位姿（Z轴下移）
        target_pos = current_tf.get_matrix()[0, :3, 3].clone()
        target_pos[1] -= delta_z  # 假设Z轴垂直向下
        # # target_pos[2] -= 0.1
        target_rot = current_tf.get_matrix()[0, :3, :3]
        print(f'current pos :{target_pos}')
        print(f'current rot :{target_rot}')
        
        # 转换为Transform3D
        target_tf = pk.Transform3d(
            pos=target_pos,
            rot=target_rot,
            device=self.device
        )

        # 求解逆运动学
        ik_result = self.ik_solver.solve(target_tf)
        end_time1 = time.time()
        print(f'ik compute time {(start_time - end_time1)*1000}')
        # print(ik_result.solutions)
        # finalik_list = []
        best_ik = None
        current_cost = 38.0
        current_q_np = self.current_q.clone().cpu().numpy()
        for iksolution in ik_result.solutions[0] :
            # ik_tf = self.chain.forward_kinematics(iksolution)
            # ik_pos = ik_tf.get_matrix()[0, :3, 3].clone()
            # ik_pos_np = ik_pos.cpu().numpy()
            # target_pos_np = target_pos.cpu().numpy()
            # # 计算欧氏距离
            # distance = np.linalg.norm(ik_pos_np - target_pos_np)
            # if distance <= 0.001:
            # finalik_list.append(iksolution)
            iksolution_np = iksolution.clone().cpu().numpy()
            engary_cost = np.linalg.norm(iksolution_np - current_q_np)
            if engary_cost < current_cost :
                current_cost = engary_cost
                best_ik = iksolution

        end_time2 = time.time()
        print(f'chose ik time {(end_time1 - end_time2)*1000}')       
        
        if not best_ik is None:
            self._publish_joint_command(best_ik)
        else :
            rospy.logerr("IK求解失败")
        # if ik_result.converged.any():
        #     #for qpos in ik_result.solutions[0] :
        #     self._publish_joint_command(ik_result.solutions[0][0])
        # else:
        #     rospy.logerr("IK求解失败")

  
    
    def _publish_joint_command(self, q_target):
        """发布关节指令"""
        # msg = JointState()
        # msg.header.stamp = rospy.Time.now()
        # msg.name = self.controlled_joints
        # msg.position = q_target.cpu().numpy().tolist()
        # self.joint_command_pub.publish(msg)
        
        goal = kinova_msgs.msg.ArmJointAnglesGoal()
        
        pi = math.pi
        
        q_target = q_target.cpu().numpy()
        print(f'set qpos : {q_target}')
        goal.angles.joint1 = q_target[0] * 360 /(2*pi)
        goal.angles.joint2 = q_target[1] * 360 /(2*pi) 
        goal.angles.joint3 = q_target[2] * 360 /(2*pi)
        goal.angles.joint4 = q_target[3] * 360 /(2*pi)
        goal.angles.joint5 = q_target[4] * 360 /(2*pi) 
        goal.angles.joint6 = q_target[5] * 360 /(2*pi)
        #goal.angles.joint7 = q_target[6] * 360 /pi
        
        rospy.loginfo(f"发布关节指令: {goal.angles}")
        self.client.send_goal(goal)
        if self.client.wait_for_result(rospy.Duration(20.0)):
            return self.client.get_result()
        else:
            print(' the joint angle action timed-out')
            self.client.cancel_all_goals()
            return None

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        # 自动选择设备：cuda可用时使用GPU，否则使用CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        controller = KinovaIKController(device=device)
        controller.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("节点已终止")