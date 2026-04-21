"""
Main teleoperation controller for dual Panda + Allegro.
Standalone MuJoCo mode: no ROS2 dependency.

Pipeline per tick:
  1) Pull latest tracking sample (wrists + 25 finger keypoints per hand).
  2) DexPilot-retarget the finger keypoints -> 16 Allegro joint angles per hand.
  3) Freeze hand joints in a kinematic mink copy, compose EE pose targets
     from wrist deltas relative to a calibrated reference pose.
  4) Solve bimanual IK with mink to get 7 Panda joint targets per arm.
  5) Low-pass filter commands and apply to MuJoCo actuators.

Controls:
  'C' — Calibrate: capture current tracking + robot EE poses as origin.
  'S' — Toggle engage/disengage (auto-calibrates if not yet calibrated).
"""
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
import mujoco
import mujoco.viewer
import mink

from avp_teleop.avp_interface import HandTrackingSource
from avp_teleop.hand_retargeter import HandRetargeter


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'multipanda_ros2', 'franka_description', 'mujoco', 'franka',
    'mj_dual_allegro_scene.xml'
)

_DEFAULT_DEX_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dex-urdf', 'robots', 'hands'
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# Rotation from AVP (Y-up, headset origin) to MuJoCo world (Z-up).
# Operator faces +X of the robot workspace.
_R_AVP_TO_MJ = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0,  1.0,  0.0],
], dtype=float)


def _se3_inv(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform without np.linalg.inv."""
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


# Panda home joint angles (typical "ready" pose)
_Q_REST = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


# ---------------------------------------------------------------------------
# Per-arm runtime state
# ---------------------------------------------------------------------------

@dataclass
class _ArmState:
    """Per-arm runtime state: calibration, IK task, filtered commands."""
    side: str                                       # 'left' or 'right'
    avp_side: str                                   # which AVP hand drives this arm

    # Joint addresses in the model's qpos vector
    panda_joint_names: list = field(default_factory=list)
    hand_joint_names: list = field(default_factory=list)
    panda_qpos_adr: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    hand_qpos_adr: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # mink FrameTask targeting the EE site
    frame_task: Optional[mink.FrameTask] = None

    # DexPilot hand retargeter
    hand_retargeter: Optional[HandRetargeter] = None

    # Calibration references
    T_wrist_ref: Optional[np.ndarray] = None        # AVP wrist at calibration
    T_ee_ref: Optional[np.ndarray] = None            # Robot EE at calibration

    # Filtered commands
    q_arm_cmd: Optional[np.ndarray] = None
    q_arm_vel_cmd: Optional[np.ndarray] = None
    q_hand_cmd: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class StandaloneTeleopController:
    """Standalone teleoperation controller using mink IK + DexPilot retargeting.

    Connects a HandTrackingSource (mock or AVP) to dual Panda arms with
    Allegro hands. Runs physics + visualization in MuJoCo viewer.

    Press 'C' to calibrate (capture reference poses), then 'S' to engage.
    While engaged the robots track the operator's hands. While disengaged
    the robots hold their last commanded position.
    """

    def __init__(self, model_path: str = None, source: HandTrackingSource = None,
                 control_freq: float = 100.0, dex_urdf_dir: str = None):
        if model_path is None:
            model_path = _DEFAULT_MODEL
        if dex_urdf_dir is None:
            dex_urdf_dir = _DEFAULT_DEX_URDF

        print(f"Loading model: {model_path}")
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._dt = 1.0 / control_freq

        # Source
        self._source = source
        self._needs_avp_transform = False

        # Engagement state (modified from key callback, consumed in main loop)
        self._engaged = False
        self._calibrated = False
        self._calibrate_requested = False
        self._toggle_engage_requested = False

        # ---- mink configuration (separate kinematic copy for IK) ----
        self._configuration = mink.Configuration(self._model)

        # ---- Build per-arm state ----
        self._arms: Dict[str, _ArmState] = {}
        for side, avp_side in [('left', 'left'), ('right', 'right')]:
            arm = _ArmState(side=side, avp_side=avp_side)

            arm.panda_joint_names = [f'mj_{side}_joint{i}' for i in range(1, 8)]
            arm.hand_joint_names = [f'mj_{side}_allegro_joint_{i}_0' for i in range(16)]
            arm.panda_qpos_adr = np.array(
                [self._model.joint(j).qposadr[0] for j in arm.panda_joint_names],
                dtype=int,
            )
            arm.hand_qpos_adr = np.array(
                [self._model.joint(j).qposadr[0] for j in arm.hand_joint_names],
                dtype=int,
            )

            arm.frame_task = mink.FrameTask(
                frame_name=f'mj_{side}_ee_site',
                frame_type='site',
                position_cost=1.0,
                orientation_cost=0.3,
                lm_damping=5.0,
            )

            arm.hand_retargeter = HandRetargeter(
                side=side, dex_urdf_dir=dex_urdf_dir,
            )

            self._arms[side] = arm

        # ---- Actuator maps ----
        self._actuator_map = {}
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self._actuator_map[name] = i

        self._arm_act_ids = {
            s: [self._actuator_map[f'mj_{s}_act_pos{i}'] for i in range(1, 8)]
            for s in ('left', 'right')
        }
        self._hand_act_ids = {
            s: [self._actuator_map[f'mj_{s}_allegro_act_joint_{i}_0'] for i in range(16)]
            for s in ('left', 'right')
        }

        # ---- Shared mink tasks and limits ----
        self._posture = mink.PostureTask(self._model, cost=5e-2)
        self._tasks = [
            self._arms['left'].frame_task,
            self._arms['right'].frame_task,
            self._posture,
        ]

        vel_limits = {}
        for side in ('left', 'right'):
            for j in self._arms[side].panda_joint_names:
                vel_limits[j] = 1.0  # rad/s
        self._limits = [
            mink.ConfigurationLimit(self._model),
            mink.VelocityLimit(self._model, vel_limits),
        ]

        # ---- Smoothing / safety parameters ----
        self._smooth_tau_arm = 0.20       # seconds (arm low-pass time constant)
        self._smooth_tau_hand = 0.03      # seconds (hand low-pass time constant)
        self._accel_limit = 8.0           # rad/s^2 per joint
        self._vel_limit = 1.0             # rad/s per joint
        self._ws_box_lo = np.array([-0.7, -0.9, 0.3])  # workspace safety box
        self._ws_box_hi = np.array([ 0.9,  0.9, 1.8])

        # ---- Initialize from keyframe ----
        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)

        q_init = self._data.qpos.copy()
        self._configuration.update(q_init)
        self._posture.set_target_from_configuration(self._configuration)

        # Seed per-arm commands from initial state
        for arm in self._arms.values():
            arm.q_arm_cmd = self._configuration.q[arm.panda_qpos_adr].copy()
            arm.q_arm_vel_cmd = np.zeros(7)
            arm.q_hand_cmd = self._configuration.q[arm.hand_qpos_adr].copy()

        # Initialize all actuator ctrl to match current joint positions
        for side in ('left', 'right'):
            arm = self._arms[side]
            for i, act_id in enumerate(self._arm_act_ids[side]):
                self._data.ctrl[act_id] = arm.q_arm_cmd[i]
            for i, act_id in enumerate(self._hand_act_ids[side]):
                self._data.ctrl[act_id] = arm.q_hand_cmd[i]

        # Physics substeps
        self._n_substeps = max(1, round(self._dt / self._model.opt.timestep))

        # Pre-settle physics (avoid oscillation on first frames)
        for _ in range(2000):
            mujoco.mj_step(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        self._home_qpos = self._data.qpos.copy()
        self._home_ctrl = self._data.ctrl.copy()

        print(f"Controller initialized (mink IK + DexPilot hand retargeting)")
        print(f"  Control rate: {control_freq} Hz, substeps: {self._n_substeps}")
        print(f"  Press 'C' to calibrate, 'S' to engage/disengage")

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def set_source(self, source: HandTrackingSource):
        """Set the tracking source (needed for ManoMockHuman which requires model/data)."""
        from avp_teleop.avp_streamer import AVPStreamer
        self._source = source
        self._needs_avp_transform = isinstance(source, AVPStreamer)

    # ------------------------------------------------------------------
    # Coordinate transform
    # ------------------------------------------------------------------

    def _compose_ee_target(self, arm: _ArmState, T_wrist_now: np.ndarray) -> np.ndarray:
        """Compute the 4x4 robot EE target from wrist delta since calibration."""
        # Wrist delta in source frame
        dT = T_wrist_now @ _se3_inv(arm.T_wrist_ref)

        # Convert delta to MuJoCo frame if source is AVP
        if self._needs_avp_transform:
            R4 = np.eye(4)
            R4[:3, :3] = _R_AVP_TO_MJ
            dT = R4 @ dT @ _se3_inv(R4)

        # Apply delta to robot EE reference
        T_target = dT @ arm.T_ee_ref

        # Workspace safety clamp (translation only)
        T_target[:3, 3] = np.clip(
            T_target[:3, 3], self._ws_box_lo, self._ws_box_hi)

        return T_target

    # ------------------------------------------------------------------
    # Calibration & engagement
    # ------------------------------------------------------------------

    def _calibrate(self):
        """Capture current tracking + robot EE poses as the engagement origin."""
        if self._source is None:
            print("[CALIBRATE] No source connected.")
            return

        tracking = self._source.get_latest()

        bad = []
        for side, arm in self._arms.items():
            T_wrist = tracking[f'{arm.avp_side}_wrist']
            t = T_wrist[:3, 3]
            if not np.isfinite(T_wrist).all() or float(np.linalg.norm(t)) < 1e-6:
                bad.append(f"{side}({arm.avp_side})")

        if bad:
            print(f"[CALIBRATE] Warning: wrists not tracked: {', '.join(bad)}. "
                  "Put both hands in view and retry.")
            return

        mujoco.mj_forward(self._model, self._data)
        for side, arm in self._arms.items():
            arm.T_wrist_ref = tracking[f'{arm.avp_side}_wrist'].copy()

            site_name = f'mj_{side}_ee_site'
            site_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            T_ee = np.eye(4)
            T_ee[:3, :3] = self._data.site_xmat[site_id].reshape(3, 3).copy()
            T_ee[:3, 3] = self._data.site_xpos[site_id].copy()
            arm.T_ee_ref = T_ee

        self._calibrated = True
        print("[CALIBRATED] Reference poses captured. Press 'S' to engage.")

    def _key_callback(self, keycode):
        """Handle viewer key events.

        Only sets flags here — actual work is done in the main loop to avoid
        thread-safety issues (viewer callback runs on the render thread while
        mj_step runs on the main thread).
        """
        if keycode == ord('C'):
            self._calibrate_requested = True
        elif keycode == ord('S'):
            self._toggle_engage_requested = True

    def _process_key_requests(self):
        """Process deferred key requests on the main thread (safe for MuJoCo)."""
        if self._calibrate_requested:
            self._calibrate_requested = False
            self._calibrate()

        if self._toggle_engage_requested:
            self._toggle_engage_requested = False
            if not self._calibrated:
                self._calibrate()
                if not self._calibrated:
                    return
            self._engaged = not self._engaged
            if self._engaged:
                print("[ENGAGED] Tracking active. Press 'S' to stop.")
            else:
                print("[DISENGAGED] Robots holding position. Press 'S' to engage.")

    # ------------------------------------------------------------------
    # Control step
    # ------------------------------------------------------------------

    def step(self):
        """Execute one control step: read tracking, retarget, IK, apply, physics."""
        if self._source is None:
            for _ in range(self._n_substeps):
                mujoco.mj_step(self._model, self._data)
            return

        tracking = self._source.get_latest()

        if not (self._engaged and self._calibrated):
            for _ in range(self._n_substeps):
                mujoco.mj_step(self._model, self._data)
            return

        # 1) Hand retargeting (DexPilot)
        hand_q: Dict[str, np.ndarray] = {}
        for side, arm in self._arms.items():
            T_wrist = tracking[f'{arm.avp_side}_wrist']
            fingers = tracking[f'{arm.avp_side}_fingers']
            hand_q[side] = arm.hand_retargeter.retarget(T_wrist, fingers)

        # 2) Freeze retargeted hand joints in the mink kinematic copy so IK
        #    computes Jacobians with the hands in their commanded pose.
        q_full = self._configuration.q.copy()
        for side, arm in self._arms.items():
            q_full[arm.hand_qpos_adr] = hand_q[side]
        self._configuration.update(q_full)

        # 3) Compose EE targets from wrist deltas and set frame tasks
        for side, arm in self._arms.items():
            T_wrist = tracking[f'{arm.avp_side}_wrist']
            T_target = self._compose_ee_target(arm, T_wrist)
            arm.frame_task.set_target(mink.SE3.from_matrix(T_target))

        # Keep posture target near q_rest (nullspace regularizer)
        q_post = self._configuration.q.copy()
        for side, arm in self._arms.items():
            q_post[arm.panda_qpos_adr] = _Q_REST
        self._posture.set_target(q_post)

        # 4) Solve bimanual IK
        ik_ok = True
        try:
            vel = mink.solve_ik(
                self._configuration, self._tasks, self._dt,
                solver="daqp", damping=5.0, limits=self._limits,
            )
            self._configuration.integrate_inplace(vel, self._dt)
        except Exception:
            ik_ok = False

        # 5) Smooth commands and apply to actuators
        alpha_arm = 1.0 - np.exp(-self._dt / max(1e-6, self._smooth_tau_arm))
        alpha_hand = 1.0 - np.exp(-self._dt / max(1e-6, self._smooth_tau_hand))
        v_max = float(self._vel_limit)
        a_max = float(self._accel_limit)

        for side, arm in self._arms.items():
            # Hand: always tracks (not gated on IK success)
            arm.q_hand_cmd = arm.q_hand_cmd + alpha_hand * (hand_q[side] - arm.q_hand_cmd)
            for i, act_id in enumerate(self._hand_act_ids[side]):
                self._data.ctrl[act_id] = arm.q_hand_cmd[i]

            # Arm: only advances if IK solved
            if ik_ok:
                q_arm_new = self._configuration.q[arm.panda_qpos_adr].copy()
                q_smoothed = arm.q_arm_cmd + alpha_arm * (q_arm_new - arm.q_arm_cmd)
                v_des = (q_smoothed - arm.q_arm_cmd) / self._dt
                v_des = np.clip(v_des, -v_max, v_max)
                dv_max = a_max * self._dt
                v_new = arm.q_arm_vel_cmd + np.clip(
                    v_des - arm.q_arm_vel_cmd, -dv_max, dv_max)
                arm.q_arm_cmd = arm.q_arm_cmd + v_new * self._dt
                arm.q_arm_vel_cmd = v_new
            else:
                # IK failed — decay velocity to zero under accel limit
                dv_max = a_max * self._dt
                arm.q_arm_vel_cmd = arm.q_arm_vel_cmd + np.clip(
                    -arm.q_arm_vel_cmd, -dv_max, dv_max)

            for i, act_id in enumerate(self._arm_act_ids[side]):
                self._data.ctrl[act_id] = arm.q_arm_cmd[i]

        # Step physics
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Run the simulation with MuJoCo viewer."""
        print("Starting teleoperation. Close the MuJoCo viewer to stop.")
        print("Press 'C' to calibrate, then 'S' to engage.")

        self._data.ctrl[:] = self._home_ctrl[:]
        mujoco.mj_forward(self._model, self._data)

        with mujoco.viewer.launch_passive(
            self._model, self._data,
            key_callback=self._key_callback
        ) as viewer:
            self._data.ctrl[:] = self._home_ctrl[:]

            start_time = time.time()
            step_count = 0

            while viewer.is_running():
                step_start = time.time()

                # Detect viewer reset (backspace) and restore home state
                if self._data.time == 0.0 and step_count > 0:
                    self._data.qpos[:] = self._home_qpos[:]
                    self._data.qvel[:] = 0
                    self._data.ctrl[:] = self._home_ctrl[:]
                    mujoco.mj_forward(self._model, self._data)
                    # Reset state
                    self._engaged = False
                    self._calibrated = False
                    q_init = self._home_qpos.copy()
                    self._configuration.update(q_init)
                    for arm in self._arms.values():
                        arm.T_wrist_ref = None
                        arm.T_ee_ref = None
                        arm.q_arm_cmd = self._configuration.q[arm.panda_qpos_adr].copy()
                        arm.q_arm_vel_cmd = np.zeros(7)
                        arm.q_hand_cmd = self._configuration.q[arm.hand_qpos_adr].copy()
                    print("[RESET] Viewer reset. Press 'C' to calibrate.")

                self._process_key_requests()
                self.step()
                viewer.sync()
                step_count += 1

                # Rate limiting
                elapsed = time.time() - step_start
                sleep_time = self._dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Status print every 5 seconds
                if step_count % int(5 / self._dt) == 0:
                    total_elapsed = time.time() - start_time
                    actual_freq = step_count / total_elapsed
                    state = "ENGAGED" if self._engaged else "DISENGAGED"
                    cal = "CAL" if self._calibrated else "UNCAL"
                    print(f"[{total_elapsed:.0f}s] {state}/{cal} freq={actual_freq:.1f}Hz")

        print("Viewer closed. Stopping.")

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data
