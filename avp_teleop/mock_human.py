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
  - Little tip (gray) — visualized but ignored (Allegro has no pinky)
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from avp_teleop.avp_interface import HandTrackingSource

_FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'little']


class ManoMockHuman(HandTrackingSource):
    """Interactive mock using draggable mocap bodies in the MuJoCo viewer.

    Reads mocap_pos/mocap_quat from the MuJoCo data for 12 bodies (6 per hand).
    Returns wrist poses and fingertip positions in world frame.

    When the wrist is dragged, all fingertip markers follow as a rigid body
    so the whole hand moves together.

    Must be created after the MuJoCo model is loaded, since it needs model/data refs.
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
        """Get world position of a mocap body."""
        return self._data.mocap_pos[self._mocap_ids[name]].copy()

    def _get_mocap_quat(self, name: str) -> np.ndarray:
        """Get quaternion [w,x,y,z] of a mocap body."""
        return self._data.mocap_quat[self._mocap_ids[name]].copy()

    @staticmethod
    def _mj_quat_to_scipy(q):
        """Convert MuJoCo [w,x,y,z] to scipy [x,y,z,w]."""
        return np.array([q[1], q[2], q[3], q[0]])

    def sync_fingertips_to_wrist(self):
        """Make all fingertip mocap markers follow the wrist as a rigid body.

        Call this each frame BEFORE reading tracking data. When the user drags
        the wrist marker, all fingertip markers move with it (translation +
        rotation). Individually dragged fingertips are unaffected because the
        wrist delta will be zero on those frames.
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

                    # Rigid-body transform: rotate offset around wrist, then translate
                    relative = fpos - prev_pos
                    new_relative = r_delta.apply(relative)
                    self._data.mocap_pos[mid] = curr_pos + new_relative

            # Always update tracking state so next delta is from latest frame
            self._prev_wrist[side]['pos'] = curr_pos
            self._prev_wrist[side]['quat'] = curr_quat

    def get_latest(self) -> dict:
        """Read current mocap body states.

        Automatically syncs fingertip markers to follow wrist movement first.

        Returns dict with:
            left_wrist_pos: (3,) world position
            left_wrist_quat: (4,) [w,x,y,z] quaternion
            left_fingertips: dict of 'thumb','index','middle','ring' -> (3,) world positions
            right_wrist_pos, right_wrist_quat, right_fingertips: same for right hand
        """
        # Sync fingertips to follow any wrist movement before reading
        self.sync_fingertips_to_wrist()

        result = {}
        for side in ['left', 'right']:
            result[f'{side}_wrist_pos'] = self._get_mocap_pos(f'mano_{side}_wrist')
            result[f'{side}_wrist_quat'] = self._get_mocap_quat(f'mano_{side}_wrist')
            result[f'{side}_fingertips'] = {
                'thumb': self._get_mocap_pos(f'mano_{side}_thumb_tip'),
                'index': self._get_mocap_pos(f'mano_{side}_index_tip'),
                'middle': self._get_mocap_pos(f'mano_{side}_middle_tip'),
                'ring': self._get_mocap_pos(f'mano_{side}_ring_tip'),
            }
        return result

    def is_connected(self) -> bool:
        return True
