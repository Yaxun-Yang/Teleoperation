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

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/diffusion_policy')
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
from visualizer import RawScene
import cv2
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

import copy
import h5py
from robohive.utils.quat_math import euler2quat, euler2mat, mat2quat, diffQuat, mulQuat
from robohive.utils.inverse_kinematics import IKResult, qpos_from_site_pose
from robohive.robot import robot
import my_env
import torch
import hydra
import dill
import matplotlib.pyplot as plt
import os
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
    checkpoint = '../data/action2_new_eightyK.ckpt'
    hfile = '../data/rerun/action2_new_hand_over4.hdf5'
    device = 'cuda:0'

    recordinig = defaultdict(lambda: [])

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir='data')
    workspace.load_payload(payload)

    mat_to_quat = RotationTransformer('matrix', 'quaternion')
    six_to_quat = RotationTransformer('rotation_6d', 'quaternion')

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    visualizer = RawScene()
   
    env_args = {'is_hardware': True, 
                'config_path': './my_env/franka_robotiq.config',
                }

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
    episode_idx = 0

    visualizer.reset()
    try:
        while True:
            print(episode_idx)

            # Reset
            reset_noise = reset_noise*np.random.uniform(low=-1, high=1, size=env.init_qpos.shape)
            env.reset(reset_qpos=env.init_qpos+reset_noise, blocking=False)
            _, _, _, obs = env.forward()
            # Reset goal site back to nominal position
            env.sim.model.site_pos[goal_sid] = env.sim.data.site_xpos[teleop_sid]
            env.sim.model.site_quat[goal_sid] = mat2quat(np.reshape(env.sim.data.site_xmat[teleop_sid], [3,-1]))

            act = np.zeros(env.action_space.shape)
            gripper_state = 1

            target_pos = env.sim.model.site_pos[goal_sid]
            target_quat =  env.sim.model.site_quat[goal_sid]

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

            _, _, _, obs = env.step(normalized_action)

            # recover init state
            
            past_obs = obs

            step_idx = 0

            
            while True:
                if step_idx < 10:
                    time.sleep(0.1)
                    step_idx += 1
                    continue
                
                model_input = {}

                agentview_image = torch.stack([
                    torch.tensor(cv2.resize(past_obs['visual_dict']['rgb:right_cam:480x640:2d'][:, 80:-80], (92, 92))).permute(-1, 0, 1) / 255,
                    torch.tensor(cv2.resize(obs['visual_dict']['rgb:right_cam:480x640:2d'][:, 80:-80], (92, 92))).permute(-1, 0, 1) / 255
                ])

                # eye_in_hand_image = torch.stack([
                #     torch.tensor(cv2.resize(past_obs['visual_dict']['rgb:wrist_cam:480x640:2d'][:, 80:-80], (240, 240))).permute(-1, 0, 1) / 255,
                #     torch.tensor(cv2.resize(obs['visual_dict']['rgb:wrist_cam:480x640:2d'][:, 80:-80], (240, 240))).permute(-1, 0, 1) / 255
                # ])

                model_input['agentview_image'] = agentview_image
                # model_input['eye_in_hand_image'] = eye_in_hand_image
                model_input['eef_pos'] = np.stack([past_obs['obs_dict']['pos_ee'], obs['obs_dict']['pos_ee']])

                eef_quat = torch.stack([mat_to_quat.forward(torch.tensor(past_obs['obs_dict']['rot_ee'].reshape(3,3)[None]))[0],
                                    mat_to_quat.forward(torch.tensor(obs['obs_dict']['rot_ee'].reshape(3,3)[None]))[0]])
                
                model_input['eef_quat'] = eef_quat
                model_input['gripper_qpos'] = torch.stack([torch.tensor(past_obs['obs_dict']['qp_ee'][None]), torch.tensor(obs['obs_dict']['qp_ee'][None])])

                # udpate desired pos
                with torch.no_grad():
                    model_input = {obs: model_input[obs][None] for obs in model_input}
                    start_inference = time.perf_counter()
                    action_chunk = policy.predict_action(model_input)['action']
                    print(f"Inference time: {time.perf_counter()-start_inference}")
                action_chunk = action_chunk[0]
                start_execution = time.perf_counter()
                for action in action_chunk:
                    target_pos = action[:3].cpu().numpy()
                    target_quat = six_to_quat.forward(torch.tensor(action[3:9])[None])[0].cpu().detach().numpy()
                    gripper_state = action[9].cpu().numpy()
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
                            print(f"Non normalized action: {act}")
                            print(f"Normalized action: {normalized_action}")

                    episode_data = {
                        'actions': np.concatenate([target_pos, target_quat, [gripper_state]]),
                        'obs/time': obs['obs_dict']['time'],
                        'obs/arm_qpos': obs['obs_dict']['qp_arm'],
                        'obs/eef_pos': obs['obs_dict']['pos_ee'],
                        'obs/eef_rot': obs['obs_dict']['rot_ee'],
                        'obs/gripper_qpos': obs['obs_dict']['qp_ee'],
                        'obs/agentview_image': obs['visual_dict']['rgb:right_cam:480x640:2d'][:, 80:-80],
                        # 'obs/eye_in_hand_image': obs['visual_dict']['rgb:wrist_cam:480x640:2d'][:, 80:-80],
                    }
                
                    visualizer.log(episode_data)

                    # record rerun data
                    for k in episode_data:
                        recordinig[k].append(episode_data[k])

                    
                    # input('Press Enter to execute...')
                    
                    past_obs = obs
                    _, _, _, obs = env.step(normalized_action)

                    # Detect jumps
                    qpos_now = obs['obs_dict']['qp_arm']
                    qpos_arm_err = np.linalg.norm(ik_result.qpos[:7]-qpos_now[:7])
                    if qpos_arm_err>0.5:
                        print("Jump detechted. Joint error {}. This is likely caused when hardware detects something unsafe. Resetting goal to where the arm curently is to avoid sudden jumps.".format(qpos_arm_err))
                        # Reset goal back to nominal position
                        env.sim.model.site_pos[goal_sid] = env.sim.data.site_xpos[teleop_sid]
                        env.sim.model.site_quat[goal_sid] = mat2quat(np.reshape(env.sim.data.site_xmat[teleop_sid], [3,-1]))

                    step_idx += 1
                
                print(f"Execution time: {time.perf_counter()-start_execution}")
                print(step_idx)#
                # if step_idx >= 300:
                #     break
    except Exception as e:
        print(e)
        with h5py.File(hfile, 'w') as f:
            for k in recordinig:
                f[f'data/demo_1/{k}'] = np.array(recordinig[k])
        # recordinig.close()
    
    

    
    

if __name__ == '__main__':
    main()