# ROB804 — Apple Vision Pro Teleoperation for Dual Panda + Allegro

Real-time teleoperation system that maps Apple Vision Pro (AVP) hand tracking to dual Franka Panda arms with Allegro dexterous hands, using a standalone MuJoCo simulation.

## Architecture

The system runs as a single-process MuJoCo simulation with an interactive viewer. No ROS2 is required for the core teleoperation loop.

### Pipeline (per tick at 50-100 Hz)

```
AVP / Mock Source
        │
        ▼
  ┌─────────────┐
  │ Hand Retarg. │  DexPilot (dex_retargeting) — 25 AVP finger joints → 16 Allegro joints
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Arm IK     │  mink bimanual QP IK — wrist delta → 7 Panda joints per arm
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Smoothing  │  Low-pass filter + velocity/acceleration clamp
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  MuJoCo     │  PD actuators → physics step → viewer render
  └─────────────┘
```

### Module breakdown

| File | Role |
|------|------|
| `avp_teleop/teleop_controller.py` | Main loop: reads tracking, runs arm IK (mink) + hand retargeting (DexPilot), smooths commands, applies to MuJoCo actuators, steps physics, renders viewer. Handles calibration (`C` key) and engagement (`S` key). |
| `avp_teleop/hand_retargeter.py` | Maps AVP 25-joint hand skeleton to Allegro 16-joint angles using DexPilot optimization via the `dex_retargeting` library. Builds 10 inter-finger distance vectors and runs constrained optimization. |
| `avp_teleop/avp_interface.py` | Abstract base class (`HandTrackingSource`) defining the tracking data format: 4x4 wrist transforms + 25x4x4 finger keypoint arrays per hand. |
| `avp_teleop/avp_streamer.py` | Real AVP interface via `avp_stream` library (WebRTC). Returns tracking data in AVP native frame (Y-up); coordinate transform to MuJoCo Z-up is done in the controller. |
| `avp_teleop/mock_human.py` | Draggable MANO mocap bodies in the MuJoCo viewer for testing without hardware. Returns data in the same format as AVP. Double-click and drag the colored spheres. |
| `avp_teleop/test_standalone.py` | Entry point. Supports `--source mock` (interactive drag), `--source avp` (real AVP), and `--source test` (automated headless test). |
| `dex-urdf/` | Allegro hand URDF files required by `dex_retargeting` for DexPilot optimization. |
| `multipanda_ros2/franka_description/mujoco/franka/` | MuJoCo XML scene with dual Panda + Allegro + MANO mocap bodies. |

### Arm control — delta-pose tracking with mink IK

On calibration (`C` key), the controller captures the current AVP wrist pose and robot EE pose as the reference origin. Each subsequent frame computes the SE3 delta between the current and reference AVP wrist, transforms it from AVP frame (Y-up) to MuJoCo frame (Z-up), and applies it to the robot EE reference to get the target pose. `mink` solves bimanual inverse kinematics (QP with velocity and configuration limits) to find 7-DOF joint targets for each arm. A posture task pulls the arms toward a rest configuration in the IK null space.

### Hand control — DexPilot retargeting

The DexPilot algorithm (from `dex_retargeting`) takes 10 inter-finger distance vectors (pinch distances between thumb and each finger, spread distances between adjacent fingers, and palm-to-fingertip vectors) computed from the AVP 25-joint hand skeleton. It optimizes Allegro joint angles so the robot hand's fingertip vectors match the human's. This is absolute retargeting — the finger pose directly reflects the operator's hand shape, not a delta.

### Smoothing and safety

- **Low-pass filter**: First-order exponential smoothing on both arm (tau=0.20s) and hand (tau=0.03s) commands.
- **Velocity clamp**: Per-joint velocity limit of 1.0 rad/s on arm joints.
- **Acceleration clamp**: Per-joint acceleration limit of 8.0 rad/s^2.
- **Workspace box**: EE position clamped to a safety volume.
- **IK failure handling**: On IK failure the arm velocity decays to zero; hand retargeting continues independently.

## Installation

### Dependencies

```bash
pip install mujoco numpy scipy mink daqp qpsolvers

# DexPilot hand retargeting (requires CPU-only PyTorch):
pip install dex-retargeting --ignore-requires-python
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Only for real Apple Vision Pro:
pip install avp-stream
```

### dex-urdf

The `dex-urdf/` directory (containing Allegro URDF files) is included in the repository. If missing, clone it:

```bash
git clone https://github.com/dexsuite/dex-urdf
```

## Running

```bash
# Automated test (no GUI, verifies DexPilot + mink IK + physics):
python -m avp_teleop.test_standalone --source test

# Interactive with draggable MANO mocap bodies (no AVP needed):
python -m avp_teleop.test_standalone --source mock

# With real Apple Vision Pro:
python -m avp_teleop.test_standalone --source avp --avp-ip <IP>
```

### Controls (in MuJoCo viewer)

| Key | Action |
|-----|--------|
| `C` | **Calibrate** — capture current hand + robot poses as the tracking origin. Both hands must be visible. |
| `S` | **Engage / Disengage** — toggle robot tracking. Auto-calibrates if not yet calibrated. |
| `Backspace` | **Reset** — MuJoCo viewer reset. Returns robots to home pose and clears calibration. |

### Workflow

1. Launch with `--source mock` or `--source avp`
2. Position your hands (or the mocap spheres) where you want the tracking origin
3. Press `C` to calibrate
4. Press `S` to engage — robots start tracking
5. Press `S` again to disengage — robots hold position
6. Press `C` to re-calibrate from a new origin anytime

## MuJoCo Scene

Primary scene file: `multipanda_ros2/franka_description/mujoco/franka/mj_dual_allegro_scene.xml`

- Dual Panda arms with Allegro hands (`mj_left_*` / `mj_right_*`)
- MANO mocap drag bodies for interactive testing
- Keyframe "home" defines the default arm/hand pose
- Actuator naming: `mj_{left,right}_act_pos{1-7}` for arms, `mj_{left,right}_allegro_act_joint_{0-15}_0` for hands
