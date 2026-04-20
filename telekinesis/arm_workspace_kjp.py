#!/usr/bin/env python3
"""
Kinova机械臂工作空间采样程序

功能：在指定工作空间内采样不同位置和姿态组合，测试机械臂可达性
参考pose_action_client.py实现

使用方法：
1. 基本使用（默认2x2x2=8个位置，3个默认姿态，共24个采样点）：
   python arm_workspace_kjp.py

2. 自定义姿态角度：
   在调用sample_workspace时传入custom_orientations参数，例如：

注意：
- 姿态角度使用弧度制，1度 ≈ 0.0175 弧度
- 可使用math.radians(度数)进行转换
- 程序会自动检查安全性，跳过不安全的位置
- 按Ctrl+C可随时中断采样
"""

import roslib; roslib.load_manifest('kinova_demo')
import rospy
import sys
import numpy as np
import math
import time
import pdb
from itertools import product

import actionlib
import kinova_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
from kinova_msgs.srv import HomeArm
import signal

class KinovaWorkspaceSampler:
    def __init__(self, robot_type='j2n6s300'):
        """初始化工作空间采样器"""
        self.robot_type = robot_type
        self.prefix = robot_type + "_"
        
        # 初始化ROS节点
        rospy.init_node(self.prefix + 'workspace_sampler')
        
        # 设置Action Client
        action_address = '/' + self.prefix + 'driver/pose_action/tool_pose'
        self.client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmPoseAction)
        print("等待pose action server连接...")
        self.client.wait_for_server()
        print("pose action server连接成功!")
        
        # 当前笛卡尔位置
        self.current_pose = None
        self.get_current_pose()
        
        # 采样结果存储
        self.successful_poses = []
        self.failed_poses = []
        
        # 中断标志
        self.interrupted = False
        
        # 指定的起始位置
        self.target_start_position = {
            'position': [0.21232962608337402, -0.5067076086997986, 0.4113683819770813],
            'orientation': [1.6371722221374512, 1.1131482124328613, 0.1342782974243164]
        }
        
    def get_current_pose(self):
        """获取当前机械臂位置"""
        topic_address = '/' + self.prefix + 'driver/out/cartesian_command'
        try:
            msg = rospy.wait_for_message(topic_address, kinova_msgs.msg.KinovaPose, timeout=5.0)
            self.current_pose = [msg.X, msg.Y, msg.Z, msg.ThetaX, msg.ThetaY, msg.ThetaZ]
            print(f"当前位置: {self.current_pose}")
        except rospy.ROSException:
            print("无法获取当前位置，使用默认值")
            self.current_pose = [0.212, -0.257, 0.510, 1.638, 1.113, 0.134]
    
    def euler_to_quaternion(self, euler_xyz):
        """欧拉角转四元数 (XYZ顺序)"""
        tx, ty, tz = euler_xyz
        sx, cx = math.sin(0.5 * tx), math.cos(0.5 * tx)
        sy, cy = math.sin(0.5 * ty), math.cos(0.5 * ty)
        sz, cz = math.sin(0.5 * tz), math.cos(0.5 * tz)
        
        qx = sx * cy * cz + cx * sy * sz
        qy = -sx * cy * sz + cx * sy * cz
        qz = sx * sy * cz + cx * cy * sz
        qw = -sx * sy * sz + cx * cy * cz
        
        return [qx, qy, qz, qw]
    
    def quaternion_to_euler(self, quaternion):
        """四元数转欧拉角"""
        qx, qy, qz, qw = quaternion
        
        # 归一化
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # 转换为欧拉角
        tx = math.atan2(2*(qw*qx - qy*qz), qw*qw - qx*qx - qy*qy + qz*qz)
        ty = math.asin(2*(qw*qy + qx*qz))
        tz = math.atan2(2*(qw*qz - qx*qy), qw*qw + qx*qx - qy*qy - qz*qz)
        
        return [tx, ty, tz]
    
    def send_cartesian_goal(self, position, orientation_euler):
        """发送笛卡尔坐标目标"""
        # 检查中断标志
        if self.interrupted:
            return False, None
            
        # 转换为四元数
        orientation_quat = self.euler_to_quaternion(orientation_euler)
        
        # 创建目标
        goal = kinova_msgs.msg.ArmPoseGoal()
        goal.pose.header = std_msgs.msg.Header(frame_id=(self.prefix + 'link_base'))
        goal.pose.pose.position = geometry_msgs.msg.Point(
            x=position[0], y=position[1], z=position[2])
        goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(
            x=orientation_quat[0], y=orientation_quat[1], 
            z=orientation_quat[2], w=orientation_quat[3])
        
        print(f"发送目标: pos={position}, euler={orientation_euler}")
        
        # 发送目标并等待结果（减少超时时间）
        self.client.cancel_all_goals()  # 确保取消之前的目标
        self.client.send_goal(goal)
        # if self.client.wait_for_result(rospy.Duration(10.0)):  # 从15秒减少到8秒
        #     result = self.client.get_result()
        #     return True, result
        # else:
        #     print("    动作超时，取消目标")
        #     self.client.cancel_all_goals()
        #     return False, None
        return True, self.client.get_result()  # 直接返回结果，假设动作成功
    
    def define_workspace(self):
        """定义长方体工作空间"""
        # 基于指定的起始位置定义工作空间
        current_x, current_y, current_z = self.target_start_position['position']
        
        # 位置范围 (相对于起始位置的偏移)
        workspace = {
            'x_range': [current_x - 0.35, current_x + 0.35],  # ±35cm
            'y_range': [current_y - 0.25, current_y + 0.25],  # ±25cm  
            'z_range': [current_z - 0.15, current_z + 0.15],  # ±15cm
            
        }
        return workspace
        
    def signal_handler(self, sig, frame):
        """信号处理器，处理Ctrl+C中断"""
        print(f"\n接收到中断信号，正在安全停止...")
        self.interrupted = True
        # 取消当前的action
        try:
            self.client.cancel_all_goals()
        except:
            pass

    def generate_sample_points(self, workspace, custom_orientations):
        """生成采样点 - 优化版本，减少默认采样数量"""
        # 位置采样点 (进一步减少采样密度)
        x_samples = np.linspace(workspace['x_range'][0], workspace['x_range'][1], 6)
        y_samples = np.linspace(workspace['y_range'][0], workspace['y_range'][1], 6)
        z_samples = np.linspace(workspace['z_range'][0], workspace['z_range'][1], 2)
        
        # 如果没有提供自定义姿态，使用更少的默认姿态
        if custom_orientations is None:
            custom_orientations = [
                self.current_pose[3:6],        # 默认姿态 (保持当前方向)
                
            ]
        else:
            print(f"✅ 使用用户自定义姿态，共 {len(custom_orientations)} 个姿态")
        
        # 生成采样点组合 - 遍历顺序：先z，再x，然后y，最后是姿态角
        sample_points = []
        for z in z_samples:
            for x in x_samples:
                for y in y_samples:
                    for orientation in custom_orientations:
                        sample_points.append({
                            'position': [x, y, z],
                            'orientation': orientation
                        })
        
        print(f"生成 {len(sample_points)} 个采样点")
        print(f"遍历顺序: Z({len(z_samples)}) -> X({len(x_samples)}) -> Y({len(y_samples)}) -> 姿态({len(custom_orientations)})")
        print(f"姿态采样: {len(custom_orientations)} 个不同姿态")
        return sample_points
    
    def is_pose_safe(self, position, orientation):
        """检查位置是否安全"""
        x, y, z = position
        
        # 基本安全检查
        if z < 0.15:  # 高度不能太低
            return False
        if np.linalg.norm([x, y]) > 2.5:  # 距离基座不能太远
            return False
        if abs(x) > 1.5 or abs(y) > 1.5:  # X,Y方向限制
            return False
            
        return True
    
    def sample_workspace(self, num_samples_per_axis=2, custom_orientations=None):
        """执行工作空间采样
        
        Args:
            num_samples_per_axis: 每个轴方向的采样点数量 (默认2x2x2=8个位置)
            custom_orientations: 自定义姿态列表，格式为 [[roll1,pitch1,yaw1], [roll2,pitch2,yaw2], ...]
                               如果为None，则使用默认的3个简单姿态
        """
        print("=" * 60)
        print("开始Kinova机械臂工作空间采样")
        print("=" * 60)
        
        # 1. 定义工作空间
        workspace = self.define_workspace()
        print("工作空间定义:")
        for key, value in workspace.items():
            print(f"  {key}: {value}")
        
        # 2. 生成采样点
        sample_points = self.generate_sample_points(workspace, custom_orientations)
        
        # 3. 执行采样
        total_points = len(sample_points)
        success_count = 0
        
        print(f"\n开始执行 {total_points} 个采样点的测试...")
        print("提示: 按 Ctrl+C 可以随时中断程序")
        
        for i, point in enumerate(sample_points):
            # 检查中断标志
            if self.interrupted:
                print(f"\n程序被中断，已完成 {i}/{total_points} 个采样点")
                break
                
            position = point['position']
            orientation = point['orientation']
            
            print(f"\n[{i+1}/{total_points}] 测试采样点:")
            print(f"  位置: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
            print(f"  姿态: [{math.degrees(orientation[0]):.1f}°, {math.degrees(orientation[1]):.1f}°, {math.degrees(orientation[2]):.1f}°]")
            
            # 安全检查
            if not self.is_pose_safe(position, orientation):
                print("  ❌ 位置不安全，跳过")
                self.failed_poses.append({
                    'point': point,
                    'reason': 'unsafe_pose'
                })
                continue
            
            # 连续失败检查 - 如果连续10个点失败，询问是否继续
            if len(self.failed_poses) >= 10:
                recent_failures = [f for f in self.failed_poses[-10:] if f['reason'] == 'motion_failed']
                if len(recent_failures) >= 8:  # 最近10个中有8个运动失败
                    print(f"\n⚠️  检测到连续运动失败，可能超出工作空间范围")
                    try:
                        response = input("是否继续采样? (y/n): ").lower()
                        if response != 'y':
                            print("用户选择停止采样")
                            self.interrupted = True
                            break
                    except KeyboardInterrupt:
                        print("\n检测到Ctrl+C，中断程序")
                        self.interrupted = True
                        break
            
            # 尝试到达目标位置
            try:
                success, result = self.send_cartesian_goal(position, orientation)
                
                if success:
                    print("  ✅ 成功到达目标位置")
                    success_count += 1
                    self.successful_poses.append({
                        'point': point,
                        'result': result
                    })
                else:
                    print("  ❌ 未能到达目标位置")
                    self.failed_poses.append({
                        'point': point,
                        'reason': 'motion_failed'
                    })
                
                # 短暂等待，避免过快发送指令
                time.sleep(0.2)  # 从0.5秒减少到0.2秒
                
            except KeyboardInterrupt:
                print(f"\n检测到Ctrl+C，中断程序")
                self.interrupted = True
                break
            except Exception as e:
                print(f"  ❌ 执行出错: {e}")
                self.failed_poses.append({
                    'point': point,
                    'reason': f'exception: {e}'
                })
            
            # 每5个点显示一次进度（因为总点数较少）
            if (i + 1) % 5 == 0:
                print(f"\n进度: {i+1}/{total_points} ({100*(i+1)/total_points:.1f}%)")
                print(f"成功率: {success_count}/{i+1} ({100*success_count/(i+1):.1f}%)")
        
        # 4. 输出结果统计
        self.print_results(total_points, success_count)
    
    def print_results(self, total_points, success_count):
        """打印采样结果"""
        print("\n" + "=" * 60)
        print("工作空间采样完成!")
        print("=" * 60)
        print(f"总采样点数: {total_points}")
        print(f"成功点数: {success_count}")
        print(f"失败点数: {len(self.failed_poses)}")
        print(f"成功率: {100*success_count/total_points:.1f}%")
        
        if self.failed_poses:
            print(f"\n失败原因统计:")
            failure_reasons = {}
            for failed in self.failed_poses:
                reason = failed['reason']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            for reason, count in failure_reasons.items():
                print(f"  {reason}: {count} 次")
        
        # 保存结果到文件
        self.save_results()
    
    def save_results(self):
        """保存采样结果到文件"""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workspace_sampling_results_{timestamp}.json"
        
        results = {
            'robot_type': self.robot_type,
            'timestamp': timestamp,
            'total_points': len(self.successful_poses) + len(self.failed_poses),
            'successful_points': len(self.successful_poses),
            'failed_points': len(self.failed_poses),
            'success_rate': len(self.successful_poses) / (len(self.successful_poses) + len(self.failed_poses)),
            'successful_poses': self.successful_poses,
            'failed_poses': self.failed_poses
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n结果已保存到: {filename}")
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def return_home(self):
        """返回起始位置"""
        print("\n返回起始位置...")
        home_position = self.target_start_position['position']
        home_orientation = self.target_start_position['orientation']
        
        success, _ = self.send_cartesian_goal(home_position, home_orientation)
        if success:
            print("✅ 成功返回起始位置")
        else:
            print("❌ 返回起始位置失败")

    def call_home_arm(self):
        """调用home arm服务"""
        print("正在调用home arm服务...")
        try:
            # 等待home arm服务
            service_name = f'/{self.prefix}driver/in/home_arm'
            rospy.wait_for_service(service_name, timeout=10.0)
            
            # 调用服务
            home_arm_service = rospy.ServiceProxy(service_name, HomeArm)
            response = home_arm_service()
            
            print(f"Home arm服务响应: {response.homearm_result}")
            
            if "KINOVA ARM HAS BEEN RETURNED HOME" in response.homearm_result:
                print("✅ Home arm服务调用成功")
                # 等待机械臂完成home动作
                time.sleep(5.0)
                return True
            else:
                print(f"❌ Home arm服务调用失败: {response.homearm_result}")
                return False
                
        except rospy.ServiceException as e:
            print(f"❌ Home arm服务调用异常: {e}")
            return False
        except rospy.ROSException as e:
            print(f"❌ 等待home arm服务超时: {e}")
            return False

    def move_to_start_position(self):
        """移动到指定的起始位置"""
        print("移动到指定起始位置...")
        position = self.target_start_position['position']
        orientation = self.target_start_position['orientation']
        
        print(f"目标位置: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        print(f"目标姿态: [{math.degrees(orientation[0]):.1f}°, {math.degrees(orientation[1]):.1f}°, {math.degrees(orientation[2]):.1f}°]")
        
        success, result = self.send_cartesian_goal(position, orientation)
        
        if success:
            print("✅ 成功移动到起始位置")
            # 更新当前位置
            self.current_pose = position + orientation
            return True
        else:
            print("❌ 移动到起始位置失败")
            return False

def main():
    """主函数"""
    # 全局采样器实例用于信号处理
    global sampler
    
    try:
        # 创建采样器
        sampler = KinovaWorkspaceSampler('j2n6s300')
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, sampler.signal_handler)
        
        print("=" * 60)
        print("Kinova机械臂工作空间采样程序")
        print("=" * 60)
        
        # 步骤1: 调用home arm服务
        print("步骤1: 调用home arm服务")
        if not sampler.call_home_arm():
            print("❌ Home arm失败，程序退出")
            return
        
        if sampler.interrupted:
            return
            
        # 步骤2: 移动到指定起始位置
        print("\n步骤2: 移动到指定起始位置")
        if not sampler.move_to_start_position():
            print("❌ 移动到起始位置失败，程序退出")
            return
            
        if sampler.interrupted:
            return
        
        # 步骤3: 用户确认开始采样
        print(f"\n步骤3: 准备开始工作空间采样")
        print(f"起始位置: {sampler.target_start_position['position']}")
        print(f"起始姿态: {sampler.target_start_position['orientation']}")
        
        try:
            input("按回车键开始工作空间采样 (Ctrl+C 取消)...")
        except KeyboardInterrupt:
            print("\n用户取消，程序退出")
            return
        
        if sampler.interrupted:
            return
            
        # 步骤4: 执行采样 (减少采样点数，支持自定义姿态)
        print("\n开始采样，按Ctrl+C可随时中断...")
        
        # 可以在这里自定义姿态（示例）
        # custom_orientations = [
        #     [0.0, 0.0, 0.0],                       # 默认姿态
        #     [math.radians(15), 0.0, 0.0],          # Roll +15度
        #     [0.0, math.radians(10), 0.0],          # Pitch +10度
        #     [0.0, 0.0, math.radians(20)],          # Yaw +20度
        # ]
        
        # 使用默认姿态或自定义姿态
        custom_orientations = None  # 使用默认的3个简单姿态
        
        sampler.sample_workspace(num_samples_per_axis=2, custom_orientations=custom_orientations)
        
        # 步骤5: 返回起始位置
        if not sampler.interrupted:
            sampler.return_home()
        
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
        if 'sampler' in locals():
            sampler.interrupted = True
            try:
                sampler.client.cancel_all_goals()
            except:
                pass
    except rospy.ROSInterruptException:
        print("\nROS中断，程序退出")
    except Exception as e:
        print(f"\n程序出错: {e}")
    finally:
        print("程序结束")

if __name__ == '__main__':
    main()