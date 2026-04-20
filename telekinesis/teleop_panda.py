# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/robohive
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """
DESC = """
TUTORIAL: Arm+Gripper tele-op using oculus \n
    - NOTE: Tutorial is written for franka arm and robotiq gripper. This demo is a tutorial, not a generic functionality for any any environment
EXAMPLE:\n
    - python tutorials/ee_teleop.py -e rpFrankaRobotiqData-v0\n
"""
# TODO: (1) Enforce pos/rot/grip limits (b) move gripper to delta commands

# recording process usage:
# 1. make sure panda robot and gripper are launched, make sure camera is connected
# 2. run this code, press 'enter' to start the whole process
# 3. press 'A' to start recording
# 4. press 'B' to end recording
# 5. press 'X' for failure or 'Y' for success to save the recording
# 6. waiting robot reset to the initial position
# 7. repeat step 3-6

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/robohive')
import signal
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import click
import gym
import rerun as rr

from teleop.visualizer import RawScene
import cv2

import copy
import h5py
from robohive.utils.quat_math import euler2quat, euler2mat, mat2quat, diffQuat, mulQuat
from robohive.utils.inverse_kinematics import IKResult, qpos_from_site_pose
from robohive.robot import robot
import my_env
# from live_visualize import RawScene



try:
    from oculus_reader import OculusReader
except ImportError as e:
    raise ImportError("(Missing oculus_reader. HINT: Install and perform the setup instructions from https://github.com/rail-berkeley/oculus_reader)")

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

@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', default='rpFrankaRobotiqData-v1')
@click.option('-ea', '--env_args', type=str, default='', help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('-rn', '--reset_noise', type=float, default=0.0, help=('Amplitude of noise during reset'))
@click.option('-an', '--action_noise', type=float, default=0.0, help=('Amplitude of action noise during rollout'))
@click.option('-o', '--output', type=str, default="teleOp_trace.h5", help=('Output name'))
@click.option('-h', '--horizon', type=int, help='Rollout horizon', default=100)
@click.option('-n', '--num_rollouts', type=int, help='number of repeats for the rollouts', default=2)
@click.option('-f', '--output_format', type=click.Choice(['RoboHive', 'RoboSet']), help='Data format', default='RoboHive')
@click.option('-c', '--camera', multiple=True, type=str, default=[], help=('list of camera topics for rendering'))
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='Where to render?', default='none')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-gs', '--goal_site', type=str, help='Site that updates as goal using inputs', default='ee_target')
@click.option('-ts', '--teleop_site', type=str, help='Site used for teleOp/target for IK', default='end_effector')
@click.option('-ps', '--pos_scale', type=float, default=0.05, help=('position scaling factor'))
@click.option('-rs', '--rot_scale', type=float, default=0.1, help=('rotation scaling factor'))
@click.option('-gs', '--gripper_scale', type=float, default=1, help=('gripper scaling factor'))
# @click.option('-tx', '--x_range', type=tuple, default=(-0.5, 0.5), help=('x range'))
# @click.option('-ty', '--y_range', type=tuple, default=(-0.5, 0.5), help=('y range'))
# @click.option('-tz', '--z_range', type=tuple, default=(-0.5, 0.5), help=('z range'))
# @click.option('-rx', '--roll_range', type=tuple, default=(-0.5, 0.5), help=('roll range'))
# @click.option('-ry', '--pitch_range', type=tuple, default=(-0.5, 0.5), help=('pitch range'))
# @click.option('-rz', '--yaw_range', type=tuple, default=(-0.5, 0.5), help=('yaw range'))
# @click.option('-gr', '--gripper_range', type=tuple, default=(0, 1), help=('z range'))
def main(env_name, env_args, reset_noise, action_noise, output, horizon, num_rollouts, output_format, camera, seed, render, goal_site, teleop_site, pos_scale, rot_scale, gripper_scale):
    # x_range, y_range, z_range, roll_range, pitch_range, yaw_range, gripper_range):

    visualizer = RawScene()

    # seed and load environments
    env_args = {'is_hardware': True, 
                'config_path': './my_env/franka_robotiq.config',
                # 'frame_skip': 50,
                # 'model_path': '/franka_robotiq.xml', 
                # 'target_pose': np.array([0, 0, 0, 0, 0, 0, 0, 0])
                }
    
    # visualizer = RawScene()
    # time.sleep(10)

    np.random.seed(seed)
    env = gym.make(env_name, **env_args)
    env.seed(seed)
    env.env.mujoco_render_frames = False
    goal_sid = env.sim.model.site_name2id(goal_site)
    teleop_sid = env.sim.model.site_name2id(teleop_site)
    env.sim.model.site_rgba[goal_sid][3] = 1 # make visible

    def close():
        env.close()

    signal.signal(signal.SIGINT, close)

    # prep input device
    oculus_reader = OculusReader()
    pos_offset = env.sim.model.site_pos[goal_sid].copy()
    quat_offset = env.sim.model.site_quat[goal_sid].copy()
    oculus_reader_ready = False
    while not oculus_reader_ready:
        # Get the controller and headset positions and the button being pushed
        transformations, buttons = oculus_reader.get_transformations_and_buttons()
        if transformations or buttons:
            oculus_reader_ready = True
        else:
            print("Oculus reader not ready. Check that headset is awake and controller are on")
        time.sleep(0.10)

    print('Oculus Ready!')
    dataset_name = 'leap_action.hdf5'

    episode_idx = 0
    if Path(dataset_name).exists():
        with h5py.File(dataset_name, 'r') as f:
            episode_idx = len(f['data'].keys())

    # predent to know the iteration number
    gripper_iteration = 77 #!!! check this latter

    while True:
        # default actions
        print(episode_idx)

        # Reset
        reset_noise = reset_noise*np.random.uniform(low=-1, high=1, size=env.init_qpos.shape)
        env.reset(reset_qpos=env.init_qpos+reset_noise, blocking=False)

        # if gripper state is absolte value and between 1(for open) and 0(for close)
        gripper_state = 1
        # Reset goal site back to nominal position
        env.sim.model.site_pos[goal_sid] = env.sim.data.site_xpos[teleop_sid]
        env.sim.model.site_quat[goal_sid] = mat2quat(np.reshape(env.sim.data.site_xmat[teleop_sid], [3,-1]))

        # recover init state
        obs, rwd, done, env_info = env.forward()
        act = np.zeros(env.action_space.shape)
        # start rolling out

        step_idx = 0
        episode_ended = False
        recording = False
        episode_data = defaultdict(lambda: [])
        visualizer.reset()

        while True:
            if step_idx == 1:
                print('Ready')
                
            # poll input device --------------------------------------
            transformations, buttons = oculus_reader.get_transformations_and_buttons()

            # Check for reset request
            if episode_ended and buttons and (buttons['X'] or buttons['Y']):
                env.sim.model.site_pos[goal_sid] = pos_offset
                env.sim.model.site_quat[goal_sid] = quat_offset
                
                if buttons['Y']:
                    success = True
                elif buttons['X']:
                    success = False
                break
            if buttons and buttons['A']:
                if not recording:
                    print('Recording started')
                    recording = True
            if buttons and buttons['B']:
                    episode_ended = True
                    print("Episode ended")
                    time.sleep(0.5)

            # recover actions using input ----------------------------
            if transformations and 'r' in transformations:
                right_controller_pose = transformations['r']
                # VRpos, VRquat = vrfront2mj(right_controller_pose)
                VRpos, VRquat = vrbehind2mj(right_controller_pose)

                # if gripper state is relative value and using 1 for open and -1 for close, 0 for no action
                # gripper_state = 0

                # Update targets if engaged
                if buttons['RG']:
                    # dVRP/R = VRP/Rt - VRP/R0
                    dVRP = (VRpos - VRP0)
                    # dVRR = VRquat - VRR0
                    dVRR = diffQuat(VRR0, VRquat)
                    # MJP/Rt =  MJP/R0 + dVRP/R

                    rr.log('jump_trans', rr.Scalar(np.linalg.norm(dVRP)))
                    rr.log('jump_rot', rr.Scalar(np.linalg.norm(Rotation.from_quat(dVRR).as_euler('xyz')[1:])))

                    # if jump detected, not update goal
                    if np.linalg.norm(dVRP)>0.23 or np.linalg.norm(Rotation.from_quat(dVRR).as_euler('xyz')[1:])>0.9:
                        print("Jump detected. ", np.linalg.norm(dVRP), np.linalg.norm(Rotation.from_quat(dVRR).as_euler('xyz')[1:]))
                    else:
                        env.sim.model.site_pos[goal_sid] = MJP0 + dVRP
                        env.sim.model.site_quat[goal_sid] = mulQuat(MJR0, dVRR)
                    
                    # if absolte gripper control
                    if buttons['leftTrig'][0] != 0:
                        # print ('act close')
                        if gripper_state > 0:
                            gripper_state -= 1/gripper_iteration
                        else:
                            gripper_state = 0
                            print('Gripper already closed')
                    elif buttons['LG']:
                        # print('act open')
                        if gripper_state < 1:
                            gripper_state += 1/gripper_iteration
                        else:
                            gripper_state = 1
                            print('Gripper already open')
                    
                    #if relative gripper control
                    # if buttons['leftTrig'][0] != 0:
                    #     # print ('act close')
                    #     gripper_state = -1
                    # elif buttons['LG']:
                    #     # print('act open')
                    #     gripper_state = 1
                    # else:
                    #     gripper_state = 0

                # Adjust origin if not engaged
                else:
                    # RP/R0 = RP/Rt
                    MJP0 = env.sim.model.site_pos[goal_sid].copy()
                    MJR0 = env.sim.model.site_quat[goal_sid].copy()

                    # VP/R0 = VP/Rt
                    VRP0 = VRpos
                    VRR0 = VRquat

                # udpate desired pos
                target_pos = env.sim.model.site_pos[goal_sid]
                target_quat =  env.sim.model.site_quat[goal_sid]

                # Find joint space solutions
                ik_result = qpos_from_site_pose(
                            physics = env.sim,
                            site_name = teleop_site,
                            target_pos= target_pos,
                            target_quat= target_quat,
                            inplace=False,
                            regularization_strength=1.0)

                # Command robot
                if ik_result.success==False:
                    print(f"Status:{ik_result.success}, total steps:{ik_result.steps}, err_norm:{ik_result.err_norm}")
                else:
                    act[:7] = ik_result.qpos[:7]
                    act[7:] = gripper_state
                    if action_noise:
                        act = act + env.env.np_random.uniform(high=action_noise, low=-action_noise, size=len(act)).astype(act.dtype)
                    if env.normalize_act:
                        normalized_action = env.env.robot.normalize_actions(act)

                # print(f't={env.time:2.2}, a={act}, o={obs[:3]}')

                # step env using action from t=>t+1 ----------------------
                if transformations and 'r' in transformations and buttons['RG'] and not episode_ended and recording:
                # if transformations and 'r' in transformations and buttons['RG'] :
                    episode_data['actions'] += [np.concatenate([target_pos, target_quat, [gripper_state]])]
                    episode_data['dones'] += [False]
                    episode_data['rewards'] += [False]

                    episode_data['obs/time'] += [env_info['obs_dict']['time']]
                    episode_data['obs/arm_qpos'] += [env_info['obs_dict']['qp_arm']]
                    episode_data['obs/eef_pos'] += [env_info['obs_dict']['pos_ee']]
                    episode_data['obs/eef_rot'] += [env_info['obs_dict']['rot_ee']]
                    episode_data['obs/gripper_qpos'] += [env_info['obs_dict']['qp_ee']]

                    primary = cv2.resize(env_info['visual_dict']['rgb:right_cam:480x640:2d'][:, 80:-80], (92, 92))
                    # wrist = cv2.resize(env_info['visual_dict']['rgb:wrist_cam:480x640:2d'][:, 80:-80], (92, 92))
                    episode_data['obs/agentview_image'] += [primary]
                    # episode_data['obs/eye_in_hand_image'] += [wrist]

                    visualizer.log({k:v[-1] for k, v in episode_data.items()})

                cv2.imshow('primary', env_info['visual_dict']['rgb:right_cam:480x640:2d'][:, 80:-80][..., ::-1])
                cv2.waitKey(1)

                    # visualizer.log({k:v[-1] for k, v in episode_data.items()})
                
                obs, rwd, done, env_info = env.step(normalized_action)

                # Detect jumps
                qpos_now = env_info['obs_dict']['qp_arm']
                qpos_arm_err = np.linalg.norm(ik_result.qpos[:7]-qpos_now[:7])
                if qpos_arm_err>0.5:
                    print("Jump detechted. Joint error {}. This is likely caused when hardware detects something unsafe. Resetting goal to where the arm curently is to avoid sudden jumps.".format(qpos_arm_err))
                    # Reset goal back to nominal position
                    env.sim.model.site_pos[goal_sid] = env.sim.data.site_xpos[teleop_sid]
                    env.sim.model.site_quat[goal_sid] = mat2quat(np.reshape(env.sim.data.site_xmat[teleop_sid], [3,-1]))

                step_idx += 1

        print("rollout end")

        if success:
            episode_data['rewards'][-1] = 1.0
            episode_data['dones'][-1] = True
            episode_idx += 1

            with h5py.File(dataset_name, 'a') as f:
                for k in episode_data:
                    f[f'data/demo_{episode_idx}/{k}'] = np.stack(episode_data[k])
            
            print(f"Episode {episode_idx -1} saved")
        
        time.sleep(0.5)
    # save and close
    env.close()

if __name__ == '__main__':
    main()