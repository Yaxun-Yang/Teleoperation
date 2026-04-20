#!/usr/bin/env python3
from datetime import datetime
import pybullet as p
import numpy as np
import os
import time

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry  # 修改：Odometry在nav_msgs包中，不是sensor_msgs
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
import sys
from pathlib import Path
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
from scipy.spatial.transform import Rotation as R

# import keyboard
from pynput.keyboard import Key, Listener
import rospy
import threading

from collections import defaultdict

# from teleop.visualizer import RawScene
import h5py
import cv2
import termios
import tty

from kinova_msgs.srv import HomeArm
"""
This takes the glove data, and runs inverse kinematics and then publishes onto LEAP Hand.

Note how the fingertip positions are matching, but the joint angles between the two hands are not.  :) 

Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
"""

class KinovaNode:
    def __init__(self):
        rospy.init_node("teleoperation_node", anonymous=True)

        # 运动链初始化
        urdf_path = "../kinova-ros/kinova_description/urdf/robot.urdf"
        
        self.current_qpos = None

        # 关节配置
        self.controlled_joints = [
            "j2n6s300_joint_1",
            "j2n6s300_joint_2",
            "j2n6s300_joint_3",
            "j2n6s300_joint_4",
            "j2n6s300_joint_5",
            "j2n6s300_joint_6",
        ]

        # ROS通信
        self.joint_state_sub = rospy.Subscriber(
            "/j2n6s300_driver/out/joint_state",
            JointState,
            self.joint_state_callback,
            queue_size=1,
        )


    def joint_state_callback(self, msg):
        """关节状态回调函数"""
        self.current_qpos = {name: pos for name, pos in zip(msg.name, msg.position)}
        return

class LeapNode:
    def __init__(self):
        # Some parameters
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(
            np.zeros(16))

        # You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        try:
            self.dxl_client = DynamixelClient(motors, "/dev/ttyUSB0", 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(
                    motors, "/dev/ttyUSB1", 4000000)
                self.dxl_client.connect()
            except Exception:
                try:
                    self.dxl_client = DynamixelClient(
                        motors, "/dev/ttyUSB3", 4000000)
                    self.dxl_client.connect()
                except Exception:
                    self.dxl_client = DynamixelClient(motors, "COM13", 4000000)
                    self.dxl_client.connect()
        # Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kP, 84, 2
        )  # Pgain stiffness
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2
        )  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kI, 82, 2
        )  # Igain
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kD, 80, 2
        )  # Dgain damping
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2
        )  # Dgain damping for side to side should be a bit less
        # Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(
            len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # allegro compatibility
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # read position
    def read_pos(self):
        return self.dxl_client.read_pos()  # 16dof

    # read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()

    # read current
    def read_cur(self):
        return self.dxl_client.read_cur()


class SystemPybulletIK:
    def __init__(self):
        self.leap_node = LeapNode()
        self.kinova_node = KinovaNode()

        p.connect(p.DIRECT)
        kinova_path_src = "/home/yzwang/leapUMI/leaphandproject_ws/src/kinova-ros/kinova_description/urdf/robot.urdf"
        self.kinovaId = p.loadURDF(
            kinova_path_src,
            [0.0, 0.0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)

        self.episode_idx = 20
        current_time = datetime.now().strftime("%m%d_%H_%M")
        self.dataset_path = f"/media/Common/leapUMI-dataset/leap_action_{current_time}"
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        # Camera thread variables
        self.camera_running = True
        self.current_color_image = None
        self.current_depth_image = None
        self.current_pose = None
        self.camera_lock = threading.Lock()
        self.insta_lock = threading.Lock()
        self.pose_lock = threading.Lock()

        # Keyboard input thread variables
        self.recording = False
        self.should_save = False
        self.should_quit = False
        self.keyboard_lock = threading.Lock()
        
        # # Initialize camera first
        # self.init_camera()
        # # Start the camera thread
        # self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        # self.camera_thread.start()
        
        # Start the keyboard thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.keyboard_thread.start()

        # Initialize Insta360 camera
        self.init_insta360()

        # Initialize T265
        self.init_T265()

        print("Finish initialization")

    def init_insta360(self):
        # Subscribe to the Insta360 Air camera topic
        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            # '/image_view/output',
            '/insta360/image_raw',
            Image,
            self.insta360_callback,
            queue_size=1
        )

    def insta360_callback(self, msg):
        """Callback function for processing incoming camera images"""
        try:
            # Convert ROS image message to OpenCV image with RGB format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            # print("fisheye img shape: ", cv_image.shape)
            
            # Thread-safe update of current image
            with self.insta_lock:
                self.current_insta_image = cv_image.copy()
                
        except Exception as e:
            print(f"Camera callback error: {e}")

    def get_insta_images(self):
        with self.insta_lock:
            if self.current_insta_image is not None:
                return self.current_insta_image.copy()
            else:
                return None

    def init_T265(self):
        """Initialize T265 tracking camera subscriber"""
        # Subscribe to the T265 odometry topic
        self.traj_sub = rospy.Subscriber(
            '/camera/odom/sample',
            Odometry,
            self.T265_callback,
            queue_size=1
        )

    def T265_callback(self, msg):
        """Callback function for T265 odometry messages"""
        try:
            # Extract position from Odometry message
            position = msg.pose.pose.position
            x = position.x
            y = position.y
            z = position.z
            
            # Extract orientation (quaternion) from Odometry message
            orientation = msg.pose.pose.orientation
            qx = orientation.x
            qy = orientation.y
            qz = orientation.z
            qw = orientation.w
            
            # Update current_pose with thread-safe operation
            # Format: [x, y, z, qx, qy, qz, qw]
            with self.pose_lock:
                self.current_pose = np.array([x, y, z, qx, qy, qz, qw])
                
        except Exception as e:
            print(f"T265 callback error: {e}")
    
    def get_current_pose(self):
        """Get the current pose from T265 in a thread-safe manner"""
        with self.pose_lock:
            if self.current_pose is not None:
                return self.current_pose.copy()
            else:
                return None

    def init_camera(self):
        """Initialize the Azure Kinect camera"""
        import pyk4a
        from pyk4a import Config, PyK4A
        
        # launch rgbd camera
        self.k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_1536P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                camera_fps=pyk4a.FPS.FPS_30,
                synchronized_images_only=True,
            )
        )
        self.k4a.start()

        shorter_side = 720
        calibration = self.k4a.calibration
        self.K = calibration.get_camera_matrix(1)  # stand for color type
        
        # Get initial capture to determine downscale factor
        capture = self.k4a.get_capture()
        H, W = capture.color.shape[:2]
        self.downscale = shorter_side / min(H, W)
        self.K[:2] *= self.downscale
        
        print("Camera initialized")

    def camera_loop(self):
        """Camera thread loop - continuously captures images"""
        zfar = 2.0
        
        while self.camera_running:
            try:
                capture = self.k4a.get_capture()
                
                if capture.color is not None and capture.transformed_depth is not None:
                    H, W = capture.color.shape[:2]
                    H = int(H * self.downscale)
                    W = int(W * self.downscale)
        
                    color = capture.color[..., :3].astype(np.uint8)
                    color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)

                    # Thread-safe update of current images
                    with self.camera_lock:
                        self.current_color_image = color.copy()

            except Exception as e:
                print(f"Camera capture error: {e}")
                time.sleep(0.1)
                
            # time.sleep(0.033)  # ~30 FPS capture rate
            
    def get_current_images(self):
        """Get the current images from the camera thread"""
        with self.camera_lock:
            if self.current_color_image is not None:
                return self.current_color_image.copy()
            else:
                return None

    def keyboard_loop(self):
        """Keyboard thread loop - continuously monitors keyboard input"""
        import select
        import termios
        import tty
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setcbreak(fd)
            while not self.should_quit:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    with self.keyboard_lock:
                        if key == 'r' or key == 'R':
                            self.recording = True
                            print("-" * 10, "Start recording")
                        elif key == 'p' or key == 'P':
                            self.recording = False
                            print("-" * 10, "Stop recording")
                        elif key == 's' or key == 'S':
                            self.should_save = True
                            self.recording = False
                            print("Marking episode for save...")
                        elif key == 'q' or key == 'Q':
                            self.should_quit = True
                            print("Quit signal received")
                            
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def cleanup(self):
        """Cleanup resources when shutting down"""
        print("Cleaning up camera thread...")
        self.camera_running = False
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join(timeout=1.0)
        if hasattr(self, 'k4a'):
            self.k4a.stop()
        
        print("Cleaning up keyboard thread...")
        self.should_quit = True
        if hasattr(self, 'keyboard_thread'):
            self.keyboard_thread.join(timeout=1.0)
            
        print("Cleanup completed")

    def recording_func(self):
        episode_data = defaultdict(lambda: [])

        # first loop: for each trajectory
        while True:
            step_idx = 0
            self.dataset_name = f"{self.dataset_path}/leap_action_{self.episode_idx}.hdf5"
            if Path(self.dataset_name).exists():
                with h5py.File(self.dataset_name, "r") as f:
                    self.episode_idx = len(f["data"].keys())

            # second loop: for each step in one trajectory
            while True:
                t0 = time.time()
                
                # Check quit signal
                with self.keyboard_lock:
                    if self.should_quit:
                        break
                    is_recording = self.recording
                    should_save = self.should_save
                    self.should_save = False  # Reset save flag

                # recording dataset
                if is_recording:
                    print("=" * 10, "recording", "=" * 10)
                    hand_state = self.leap_node.read_pos()  # gripper_state
                    kinova_joint_state = self.kinova_node.current_qpos  # arm qpose

                    p.resetJointState(
                        self.kinovaId, 2, kinova_joint_state["j2n6s300_joint_1"]
                    )
                    p.resetJointState(
                        self.kinovaId, 3, kinova_joint_state["j2n6s300_joint_2"]
                    )
                    p.resetJointState(
                        self.kinovaId, 4, kinova_joint_state["j2n6s300_joint_3"]
                    )
                    p.resetJointState(
                        self.kinovaId, 5, kinova_joint_state["j2n6s300_joint_4"]
                    )
                    p.resetJointState(
                        self.kinovaId, 6, kinova_joint_state["j2n6s300_joint_5"]
                    )
                    p.resetJointState(
                        self.kinovaId, 7, kinova_joint_state["j2n6s300_joint_6"]
                    )

                    # current end effector pose
                    link_state = p.getLinkState(self.kinovaId, 9)
                    curr_pos = link_state[4]
                    curr_quat = link_state[5]

                    # Get current images from camera thread
                    # color_image = self.get_current_images()
                    insta_image = self.get_insta_images()
                    pose = self.get_current_pose()
                    episode_data["actions"] += [
                        np.concatenate([curr_pos, curr_quat, hand_state])
                    ]
                    episode_data["dones"] += [False]
                    episode_data["rewards"] += [False]
                    # Current data
                    episode_data["obs/time"] += [time.time()]
                    episode_data["obs/arm_qpos"] += [list(kinova_joint_state.values())]
                    episode_data["obs/eef_pos"] += [curr_pos]
                    episode_data["obs/eef_rot"] += [curr_quat]
                    episode_data["obs/gripper_qpos"] += [hand_state]
                    episode_data["obs/traj_pos"] += [pose[:3]]
                    episode_data["obs/traj_quat"] += [pose[3:7]]

                    # Add camera images
                    # episode_data["obs/aside_image"] += [color_image]
                    # episode_data["obs/insta_image"] += [insta_image]
                    episode_data["obs/agentview_image"] += [insta_image]

                    step_idx += 1

                # Save episode if requested
                if should_save:
                    print("You pressed 's': Success and Next Episode ...")
                    if episode_data.get("actions") is not None and len(episode_data["actions"]) > 0:
                        episode_data['rewards'][-1] = 1.0
                        episode_data['dones'][-1] = True
                        with h5py.File(self.dataset_name, 'a') as f:
                            for k in episode_data:
                                f[f'data/demo/{k}'] = np.stack(episode_data[k])
                        print(f"Episode {self.episode_idx} saved with {len(episode_data['actions'])} steps")
                        self.episode_idx += 1

                    episode_data = defaultdict(lambda: [])
                    break

                t1 = time.time()
                print(f"Step time: {t1 - t0:.3f}s")
                time.sleep(0.2 - (t1 - t0))  # Maintain ~5 Hz

            # Check quit signal again
            with self.keyboard_lock:
                if self.should_quit:
                    break

        # end while 1
        # self.cleanup()
        print(f"All {self.episode_idx} Episodes saved")

def main(args=None):

    system = SystemPybulletIK()

    system.recording_func()

if __name__ == "__main__":
    # if not call_home_arm():
    #     print("❌ Home arm失败，程序退出")
    # else:
    main()
