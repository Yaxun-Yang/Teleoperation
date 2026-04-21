"""
Mock human model for testing teleoperation without Apple Vision Pro.

ManoMockHuman: reads draggable MANO hand mocap bodies from the MuJoCo viewer.
Double-click and drag the colored spheres in the viewer to teleoperate.

Each hand has 6 drag points:
  - Wrist (green) — controls arm IK target; dragging it moves the whole hand
  - Thumb tip (red) — controls thumb joints
  - Index tip (blue) — controls index joints
  - Middle tip (yellow) — controls middle joints
  - Ring tip (orange) — controls ring joints
  - Little tip (gray) — visualized but used only for DexPilot spread vectors
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from avp_teleop.avp_interface import HandTrackingSource

_FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'little']

# AVP HandSkeleton tip indices used for finger data
_TIP_INDICES = {
    'thumb': 4,
    'index': 9,
    'middle': 14,
    'ring': 19,
    'little': 24,
}


class ManoMockHuman(HandTrackingSource):
    """Interactive mock using draggable mocap bodies in the MuJoCo viewer.

    Reads mocap_pos/mocap_quat from the MuJoCo data for 12 bodies (6 per hand).
    Returns data in the standard AVP format (4x4 wrist transforms, 25x4x4 finger
    keypoint arrays) so the same controller code path works for both mock and AVP.

    When the wrist is dragged, all fingertip markers follow as a rigid body.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model = model
        self._data = data

        # Look up mocap indices for each body
        self._mocap_ids = {}
        body_names = [
            'mano_left_wrist', 'mano_left_thumb_tip', 'mano_left_index_tip',
            'mano_left_middle_tip', 'mano_left_ring_tip', 'mano_left_little_tip',
            'mano_right_wrist', 'mano_right_thumb_tip', 'mano_right_index_tip',
            'mano_right_middle_tip', 'mano_right_ring_tip', 'mano_right_little_tip',
        ]
        for name in body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert bid >= 0, f"Mocap body '{name}' not found in model"
            mocap_id = model.body_mocapid[bid]
            assert mocap_id >= 0, f"Body '{name}' is not a mocap body"
            self._mocap_ids[name] = mocap_id

        # Track previous wrist state for rigid-body sync
        self._prev_wrist = {}
        for side in ['left', 'right']:
            wrist_mid = self._mocap_ids[f'mano_{side}_wrist']
            self._prev_wrist[side] = {
                'pos': data.mocap_pos[wrist_mid].copy(),
                'quat': data.mocap_quat[wrist_mid].copy(),
            }

    def _get_mocap_pos(self, name: str) -> np.ndarray:
        return self._data.mocap_pos[self._mocap_ids[name]].copy()

    def _get_mocap_quat(self, name: str) -> np.ndarray:
        return self._data.mocap_quat[self._mocap_ids[name]].copy()

    @staticmethod
    def _mj_quat_to_scipy(q):
        """Convert MuJoCo [w,x,y,z] to scipy [x,y,z,w]."""
        return np.array([q[1], q[2], q[3], q[0]])

    def sync_fingertips_to_wrist(self):
        """Make all fingertip mocap markers follow the wrist as a rigid body.

        Call this each frame BEFORE reading tracking data.
        """
        for side in ['left', 'right']:
            wrist_mid = self._mocap_ids[f'mano_{side}_wrist']
            curr_pos = self._data.mocap_pos[wrist_mid].copy()
            curr_quat = self._data.mocap_quat[wrist_mid].copy()

            prev_pos = self._prev_wrist[side]['pos']
            prev_quat = self._prev_wrist[side]['quat']

            pos_delta = curr_pos - prev_pos

            r_prev = Rotation.from_quat(self._mj_quat_to_scipy(prev_quat))
            r_curr = Rotation.from_quat(self._mj_quat_to_scipy(curr_quat))
            r_delta = r_curr * r_prev.inv()

            has_moved = (np.linalg.norm(pos_delta) > 1e-7
                         or r_delta.magnitude() > 1e-7)

            if has_moved:
                for finger in _FINGER_NAMES:
                    name = f'mano_{side}_{finger}_tip'
                    mid = self._mocap_ids[name]
                    fpos = self._data.mocap_pos[mid].copy()
                    relative = fpos - prev_pos
                    new_relative = r_delta.apply(relative)
                    self._data.mocap_pos[mid] = curr_pos + new_relative

            self._prev_wrist[side]['pos'] = curr_pos
            self._prev_wrist[side]['quat'] = curr_quat

    def get_latest(self) -> dict:
        """Read current mocap body states in standard AVP format.

        Returns dict with:
            head: (4,4) identity (no head tracking in mock mode)
            left_wrist / right_wrist: (4,4) wrist transforms in MuJoCo world frame
            left_fingers / right_fingers: (25,4,4) finger keypoint arrays with
                tip positions at the standard AVP indices (4,9,14,19,24)
        """
        self.sync_fingertips_to_wrist()

        result = {'head': np.eye(4)}
        for side in ['left', 'right']:
            # Build 4x4 wrist transform from mocap pos + quat
            wrist_pos = self._get_mocap_pos(f'mano_{side}_wrist')
            wrist_quat = self._get_mocap_quat(f'mano_{side}_wrist')

            T_wrist = np.eye(4)
            R_flat = np.empty(9)
            mujoco.mju_quat2Mat(R_flat, wrist_quat)
            T_wrist[:3, :3] = R_flat.reshape(3, 3)
            T_wrist[:3, 3] = wrist_pos
            result[f'{side}_wrist'] = T_wrist

            # Build (25,4,4) finger keypoints with tips at AVP indices
            fingers = np.tile(np.eye(4), (25, 1, 1))
            for finger_name, tip_idx in _TIP_INDICES.items():
                tip_pos = self._get_mocap_pos(f'mano_{side}_{finger_name}_tip')
                fingers[tip_idx, :3, 3] = tip_pos
            result[f'{side}_fingers'] = fingers

        return result

    def is_connected(self) -> bool:
        return True
