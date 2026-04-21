"""
Main teleoperation controller for dual Panda + Allegro.
Standalone MuJoCo mode: no ROS2 dependency.

Press 'S' in the viewer to toggle engagement:
  - Engaged: MANO hand movement is retargeted to the robot in real time.
    Arms use frame-to-frame delta tracking (difference between current and
    previous MANO wrist positions applied to robot EE).
    Hands use absolute retargeting: fingertip-to-palm distance from MANO
    is directly reflected to Allegro (MANO fingers shaped to Allegro lengths).
  - Disengaged: robots hold their current position; MANO hands move freely.
"""
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

from avp_teleop.arm_retargeter import ArmRetargeter
from avp_teleop.hand_retargeter import HandRetargeter
from avp_teleop.avp_interface import HandTrackingSource


# Default model path
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'multipanda_ros2', 'franka_description', 'mujoco', 'franka',
    'mj_dual_allegro_scene.xml'
)


class StandaloneTeleopController:
    """Standalone teleoperation controller using MuJoCo Python bindings.

    Connects a HandTrackingSource (mock or AVP) to dual Panda arms with Allegro hands.
    Runs the IK + retargeting + physics loop and visualizes in MuJoCo viewer.

    Press 'S' to toggle engagement. While engaged:
    - Arms: frame-to-frame delta tracking. Each frame computes the MANO wrist
      position change (in world/simulator coordinates) and applies the same
      displacement to the robot arm target (in robot base coordinates).
    - Hands: absolute retargeting. The fingertip-to-palm vector from MANO
      (in world frame) is transformed into the Allegro hand frame and used
      directly as the IK target. MANO fingertip positions are pre-shaped to
      match Allegro finger lengths so no extra scaling is needed.
    While disengaged the robots hold their last commanded position.

    Supports two source types:
    - ManoMockHuman: draggable mocap bodies in MuJoCo viewer (direct fingertip positions)
    - AVPStreamer/other: AVP-format 25-joint hand skeleton (delta-pose retargeting)
    """

    def __init__(self, model_path: str = None, source: HandTrackingSource = None,
                 control_freq: float = 100.0):
        """
        Args:
            model_path: Path to MuJoCo XML scene file.
            source: HandTrackingSource instance, or None (set later via set_source).
            control_freq: Target control loop frequency in Hz.
        """
        if model_path is None:
            model_path = _DEFAULT_MODEL

        print(f"Loading model: {model_path}")
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)

        self._source = source
        self._is_mano = False
        self._dt = 1.0 / control_freq

        # Engagement state (toggled by 'S' key in viewer)
        self._engaged = False
        # Frame-to-frame delta tracking state
        self._prev_mano_wrist = {'left': None, 'right': None}
        self._arm_target = {'left': None, 'right': None}
        self._arm_target_rot = {'left': None, 'right': None}

        # Build actuator index maps
        self._actuator_map = {}
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self._actuator_map[name] = i

        # Arm retargeters
        self._left_arm = ArmRetargeter(self._model, self._data, 'left')
        self._right_arm = ArmRetargeter(self._model, self._data, 'right')

        # Hand retargeters (scale=1.0: MANO fingertips already shaped to Allegro lengths)
        self._left_hand = HandRetargeter(hand_scale=1.0, side='left')
        self._right_hand = HandRetargeter(hand_scale=1.0, side='right')

        # Position actuator indices for arms
        self._left_arm_act_ids = [self._actuator_map[f'mj_left_act_pos{i}'] for i in range(1, 8)]
        self._right_arm_act_ids = [self._actuator_map[f'mj_right_act_pos{i}'] for i in range(1, 8)]

        # Position actuator indices for Allegro hands
        self._left_hand_act_ids = [
            self._actuator_map[f'mj_left_allegro_act_joint_{i}_0'] for i in range(16)
        ]
        self._right_hand_act_ids = [
            self._actuator_map[f'mj_right_allegro_act_joint_{i}_0'] for i in range(16)
        ]

        # EE site IDs for computing hand-local fingertip positions
        self._left_ee_site = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, 'mj_left_ee_site')
        self._right_ee_site = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, 'mj_right_ee_site')

        # Physics substeps: run multiple physics steps per control step for real-time
        self._n_substeps = max(1, round(self._dt / self._model.opt.timestep))

        # Load home keyframe (sets arms, hands, and MANO positions all at once)
        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)

        # Initialize ALL position actuator ctrl to match current joint positions.
        # This prevents unwanted drift when disengaged (PD controller target = current pos).
        for side in ['left', 'right']:
            arm = self._left_arm if side == 'left' else self._right_arm
            self._apply_arm_control(side, arm.get_joint_positions())
            hand_act_ids = self._left_hand_act_ids if side == 'left' else self._right_hand_act_ids
            for act_id in hand_act_ids:
                joint_id = self._model.actuator_trnid[act_id, 0]
                self._data.ctrl[act_id] = self._data.qpos[self._model.jnt_qposadr[joint_id]]

        # Pre-settle: run physics to let the arm reach its gravity equilibrium.
        # Without this, the PD controller causes visible oscillation in the
        # first ~2 seconds as joints settle under gravity.
        for _ in range(2000):  # 4 seconds at 0.002 timestep
            mujoco.mj_step(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        # Store settled state for viewer-reset recovery.
        # NOTE: do NOT modify m.qpos0 — MuJoCo computes kinematics as
        # Rot(axis, qpos - qpos0), so changing qpos0 shifts the reference
        # frame and makes the arm appear in the wrong configuration.
        self._home_qpos = self._data.qpos.copy()
        self._home_ctrl = self._data.ctrl.copy()

        print(f"Controller initialized:")
        print(f"  Left arm actuators: {self._left_arm_act_ids}")
        print(f"  Right arm actuators: {self._right_arm_act_ids}")
        print(f"  Left hand actuators (16): [{self._left_hand_act_ids[0]}..{self._left_hand_act_ids[-1]}]")
        print(f"  Right hand actuators (16): [{self._right_hand_act_ids[0]}..{self._right_hand_act_ids[-1]}]")
        print(f"  Physics substeps per control step: {self._n_substeps}")

    def set_source(self, source: HandTrackingSource):
        """Set the tracking source after construction (needed for ManoMockHuman)."""
        from avp_teleop.mock_human import ManoMockHuman
        self._source = source
        self._is_mano = isinstance(source, ManoMockHuman)

    def _apply_arm_control(self, side: str, joint_positions: np.ndarray):
        """Apply arm joint positions via position actuators."""
        act_ids = self._left_arm_act_ids if side == 'left' else self._right_arm_act_ids
        for i, act_id in enumerate(act_ids):
            self._data.ctrl[act_id] = joint_positions[i]

    def _apply_hand_control(self, side: str, allegro_q: np.ndarray):
        """Apply Allegro hand joint positions via position actuators."""
        act_ids = self._left_hand_act_ids if side == 'left' else self._right_hand_act_ids
        for i, act_id in enumerate(act_ids):
            self._data.ctrl[act_id] = allegro_q[i]

    def _key_callback(self, keycode):
        """Handle viewer key events. 'S' toggles engagement."""
        # GLFW key code for 'S' is ord('S') == 83
        if keycode == ord('S'):
            self._engaged = not self._engaged
            if self._engaged:
                # Reset tracking state so first engaged frame initializes them
                self._prev_mano_wrist = {'left': None, 'right': None}
                self._arm_target = {'left': None, 'right': None}
                self._arm_target_rot = {'left': None, 'right': None}
                print("[ENGAGED] Tracking MANO -> robot. Press 'S' to stop.")
            else:
                print("[DISENGAGED] Robots holding position. Press 'S' to engage.")

    def _step_mano(self):
        """Control step for ManoMockHuman source.

        Always syncs MANO hand visualization (mocap bodies move freely).
        Only updates robot controls when engaged (toggled by 'S' key).
        Arms: frame-to-frame delta — each frame computes the wrist position
              change and accumulates it into the robot arm target.
        Hands: absolute retargeting — fingertip-to-wrist offset (world frame)
               is transformed into the Allegro hand frame via EE site rotation.
        """
        tracking = self._source.get_latest()

        if not self._engaged:
            # Robots hold position; just step physics
            for _ in range(self._n_substeps):
                mujoco.mj_step(self._model, self._data)
            return

        for side in ['left', 'right']:
            arm = self._left_arm if side == 'left' else self._right_arm
            hand = self._left_hand if side == 'left' else self._right_hand
            ee_site_id = self._left_ee_site if side == 'left' else self._right_ee_site

            wrist_pos = tracking[f'{side}_wrist_pos']

            # --- ARM: frame-to-frame delta tracking (6D: position + orientation) ---
            if self._prev_mano_wrist[side] is None:
                # First engaged frame: initialise, no movement yet
                self._prev_mano_wrist[side] = wrist_pos.copy()
                ee_pos, ee_rot = arm.get_ee_pose()
                self._arm_target[side] = ee_pos.copy()
                self._arm_target_rot[side] = ee_rot.copy()
            else:
                # Compute frame delta and accumulate into target
                delta = wrist_pos - self._prev_mano_wrist[side]
                self._prev_mano_wrist[side] = wrist_pos.copy()
                self._arm_target[side] = self._arm_target[side] + delta

            arm_q = arm.solve_absolute(
                self._arm_target[side], target_rot=self._arm_target_rot[side])
            self._apply_arm_control(side, arm_q)

            # --- HAND: absolute retargeting (fingertip-to-palm in Allegro frame) ---
            # Get Allegro hand frame rotation (at EE site)
            ee_rot = self._data.site_xmat[ee_site_id].reshape(3, 3)

            fingertips_world = tracking[f'{side}_fingertips']
            fingertips_allegro = {}
            for finger, pos in fingertips_world.items():
                # Offset from MANO wrist to fingertip in world frame
                offset_world = pos - wrist_pos
                # Express in Allegro hand frame
                fingertips_allegro[finger] = ee_rot.T @ offset_world

            hand_q = hand.retarget_from_fingertips(fingertips_allegro)
            self._apply_hand_control(side, hand_q)

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

    def _step_avp(self):
        """Control step for AVP-format source (25-joint hand skeleton, delta-pose)."""
        tracking = self._source.get_latest()

        if not self._engaged:
            for _ in range(self._n_substeps):
                mujoco.mj_step(self._model, self._data)
            return

        left_arm_q = self._left_arm.solve(tracking['left_wrist'], engaged=True)
        right_arm_q = self._right_arm.solve(tracking['right_wrist'], engaged=True)

        left_hand_q = self._left_hand.retarget(tracking['left_fingers'])
        right_hand_q = self._right_hand.retarget(tracking['right_fingers'])

        self._apply_arm_control('left', left_arm_q)
        self._apply_arm_control('right', right_arm_q)
        self._apply_hand_control('left', left_hand_q)
        self._apply_hand_control('right', right_hand_q)

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

    def step(self):
        """Execute one control step: read tracking, retarget, apply commands."""
        if self._source is None:
            return

        if self._is_mano:
            self._step_mano()
        else:
            self._step_avp()

    def run(self):
        """Main loop: run simulation with viewer."""
        print("Starting teleoperation. Close the MuJoCo viewer to stop.")
        if self._is_mano:
            print("Double-click a mocap sphere and drag to move it.")
        print("Press 'S' in the viewer to toggle robot engagement.")
        print("[DISENGAGED] Robots holding position. Press 'S' to engage.")

        # Ensure ctrl is set before viewer opens (prevents first-frame drift)
        self._data.ctrl[:] = self._home_ctrl[:]
        mujoco.mj_forward(self._model, self._data)

        with mujoco.viewer.launch_passive(
            self._model, self._data,
            key_callback=self._key_callback
        ) as viewer:
            # Re-apply ctrl in case launch_passive reset data
            self._data.ctrl[:] = self._home_ctrl[:]

            start_time = time.time()
            step_count = 0

            while viewer.is_running():
                step_start = time.time()

                # Detect viewer reset (ctrl zeroed by backspace) and restore home
                if self._data.time == 0.0 and step_count > 0:
                    # Restore settled state (not keyframe — avoids re-settling)
                    self._data.qpos[:] = self._home_qpos[:]
                    self._data.qvel[:] = 0
                    self._data.ctrl[:] = self._home_ctrl[:]
                    mujoco.mj_forward(self._model, self._data)
                    self._engaged = False
                    self._prev_mano_wrist = {'left': None, 'right': None}
                    self._arm_target = {'left': None, 'right': None}
                    self._arm_target_rot = {'left': None, 'right': None}

                self.step()
                viewer.sync()

                step_count += 1

                # Rate limiting
                elapsed = time.time() - step_start
                sleep_time = self._dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Print status every 5 seconds
                if step_count % (int(5 / self._dt)) == 0:
                    total_elapsed = time.time() - start_time
                    actual_freq = step_count / total_elapsed
                    left_pos, _ = self._left_arm.get_ee_pose()
                    right_pos, _ = self._right_arm.get_ee_pose()
                    engaged_str = "ENGAGED" if self._engaged else "DISENGAGED"
                    print(f"[{total_elapsed:.0f}s] {engaged_str} freq={actual_freq:.1f}Hz "
                          f"left_ee={left_pos} right_ee={right_pos}")

        print("Viewer closed. Stopping.")

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data
