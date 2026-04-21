# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROB804 is an Apple Vision Pro (AVP) teleoperation system for dual Franka Panda arms with Allegro dexterous hands. The active development is in `avp_teleop/`, which is a standalone MuJoCo Python package (no ROS2 dependency). ROS2 integration is planned as a future step.

## Repository Structure

- **`avp_teleop/`** — Active package. Standalone MuJoCo teleoperation controller.
  - `teleop_controller.py` — Main loop: reads hand tracking, runs bimanual arm IK (mink) + hand retargeting (DexPilot), smooths commands, steps physics, renders viewer. Press 'C' to calibrate, 'S' to toggle engagement.
  - `hand_retargeter.py` — Maps AVP 25-joint hand skeleton to Allegro 16-joint angles via DexPilot optimization using the `dex_retargeting` library. Builds 10 inter-finger distance vectors as optimization target.
  - `avp_interface.py` — `HandTrackingSource` ABC defining the tracking data format.
  - `avp_streamer.py` — Real AVP interface via `avp_stream` library (WebRTC). Returns data in AVP native frame (Y-up).
  - `mock_human.py` — `ManoMockHuman`: draggable mocap bodies in MuJoCo viewer for testing without hardware. Returns data in standard AVP format.
  - `test_standalone.py` — Entry point for testing.
- **`dex-urdf/`** — Allegro hand URDF files required by `dex_retargeting` for DexPilot optimization.
- **`multipanda_ros2/`** — ROS2 Humble packages for Franka Panda (sim + real). Contains MuJoCo XML models under `franka_description/mujoco/franka/`.
- **`mujoco_ros_pkgs/`** — Fork of mujoco_ros_pkgs with ros2_control plugin.
- **`telekinesis/`** — Legacy teleoperation code (LEAP hand + Kinova, Oculus/MediaPipe). Contains a bundled `deployment/diffusion_policy/` for policy learning.

## Running

```bash
# Automated test (no GUI, verifies DexPilot + mink IK + physics):
python -m avp_teleop.test_standalone --source test

# Interactive with draggable MANO mocap bodies:
python -m avp_teleop.test_standalone --source mock

# With real Apple Vision Pro:
python -m avp_teleop.test_standalone --source avp --avp-ip <IP>
```

## Key Dependencies (avp_teleop)

- `mujoco` (Python bindings)
- `mink` (QP-based bimanual inverse kinematics)
- `dex_retargeting` (DexPilot hand retargeting algorithm)
- `torch` (CPU-only, required by dex_retargeting)
- `numpy`, `scipy`
- `daqp`, `qpsolvers` (QP solver backend for mink)
- `avp-stream` (only for real AVP hardware)

## MuJoCo Scene

Primary scene file: `multipanda_ros2/franka_description/mujoco/franka/mj_dual_allegro_scene.xml`
- Includes `mj_dual_allegro.xml` (dual Panda + Allegro), `objects.xml`, `mano_hands.xml` (mocap drag bodies)
- Keyframe "home" defines the default arm/hand pose

## Architecture Notes

- **Coordinate frames**: AVP is Y-up; MuJoCo is Z-up. The transform `_R_AVP_TO_MJ` is in `teleop_controller.py`. Applied as SE3 delta conjugation when composing arm EE targets from AVP wrist data.
- **Arm control**: Calibration-based delta-pose tracking. On calibration ('C' key), current AVP wrist + robot EE poses are captured as reference. Subsequent frames compute the wrist SE3 delta, transform it to MuJoCo frame, and apply to the robot EE reference. `mink` solves bimanual QP IK with velocity limits, configuration limits, and posture regularization.
- **Hand control**: DexPilot absolute retargeting via `dex_retargeting`. Ten inter-finger distance vectors (pinch, spread, palm-to-tip) computed from AVP 25-joint skeleton are matched to Allegro FK. Works in wrist-local frame so coordinate system doesn't matter.
- **Unified source interface**: Both `ManoMockHuman` and `AVPStreamer` return data in the same format (`{side}_wrist` as 4x4, `{side}_fingers` as 25x4x4). The controller uses a single code path for both.
- **Smoothing**: First-order low-pass on arm (tau=0.20s) and hand (tau=0.03s) commands. Acceleration clamping (8.0 rad/s^2) and velocity clamping (1.0 rad/s) on arm joints. Workspace safety box on EE translation.
- **Actuator naming convention**: `mj_{left,right}_act_pos{1-7}` for arms, `mj_{left,right}_allegro_act_joint_{0-15}_0` for hands.

## ROS2 Build (multipanda_ros2 / mujoco_ros_pkgs)

Only needed for future ROS2 integration, not for current avp_teleop work:
```bash
# Requires: ROS2 Humble, libfranka 0.9.2, Eigen 3.3.9, MuJoCo 3.2.0, dq-robotics
colcon build --packages-up-to mujoco_ros2_control_system
rosdep install --from-paths src -y --ignore-src
```
