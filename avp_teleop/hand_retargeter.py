"""
Hand retargeting: maps AVP 25-joint hand skeleton to Allegro 16-joint angles.
Uses optimization-based fingertip position matching with scipy L-BFGS-B.
"""
import numpy as np
from scipy.optimize import minimize


# Allegro hand kinematic parameters extracted from panda_with_allegro.xml
# Each finger is a chain of 4 joints. We store: base_pos, base_quat, joint_axis, link_lengths.

# Joint limits [lower, upper] for each of the 16 Allegro joints
ALLEGRO_JOINT_LIMITS = np.array([
    # Index finger (joints 0-3)
    [-0.3, 0.3],     # joint_0: abduction (Z axis)
    [-0.01, 1.6],    # joint_1: MCP flexion (Y axis)
    [-0.07, 1.86],   # joint_2: PIP flexion (Y axis)
    [-0.02, 2.01],   # joint_3: DIP flexion (Y axis)
    # Middle finger (joints 4-7)
    [-0.26, 0.26],   # joint_4: abduction
    [-0.21, 1.79],   # joint_5: MCP
    [-0.12, 1.86],   # joint_6: PIP
    [-0.21, 1.85],   # joint_7: DIP
    # Ring finger (joints 8-11)
    [-0.26, 0.29],   # joint_8: abduction
    [-0.21, 1.79],   # joint_9: MCP
    [-0.12, 1.86],   # joint_10: PIP
    [-0.21, 1.85],   # joint_11: DIP
    # Thumb (joints 12-15)
    [0.0, 1.78],     # joint_12: rotation (-X axis)
    [-0.26, 1.65],   # joint_13: MCP (Z axis)
    [-0.05, 1.85],   # joint_14: PIP (Y axis)
    [-0.09, 1.8],    # joint_15: DIP (Y axis)
])

# Allegro finger kinematic chains from the XML body hierarchy
# For each regular finger: base -> link0 (abduction, Z) -> link1 (flexion, Y) -> link2 (Y) -> link3 (Y)
# Link lengths (Z offsets between joints):
_INDEX_CHAIN = {
    'base_pos': np.array([0.0005, 0.0425, -0.005]),
    'base_quat_angle': -0.0436194,  # small tilt
    'link_offsets': [0.017, 0.0431, 0.038, 0.025],  # last is fingertip offset
    'joint_axes': ['z', 'y', 'y', 'y'],
}

_MIDDLE_CHAIN = {
    'base_pos': np.array([0.0, 0.0, -0.002]),
    'base_quat_angle': 0.0,
    'link_offsets': [0.017, 0.0431, 0.038, 0.025],
    'joint_axes': ['z', 'y', 'y', 'y'],
}

_RING_CHAIN = {
    'base_pos': np.array([0.0005, -0.0425, -0.005]),
    'base_quat_angle': 0.0436194,
    'link_offsets': [0.017, 0.0431, 0.038, 0.025],
    'joint_axes': ['z', 'y', 'y', 'y'],
}

_THUMB_CHAIN = {
    'base_pos': np.array([0.002, 0.027, -0.06342]),
    'link_offsets': [0.06874, 0.0, 0.038, 0.025],
    'joint_axes': ['-x', 'z', 'y', 'y'],
}

# Left-hand kinematic chains (Y-mirrored from right hand)
_LEFT_INDEX_CHAIN = {
    'base_pos': np.array([0.0005, -0.0425, -0.005]),
    'base_quat_angle': 0.0436194,
    'link_offsets': [0.017, 0.0431, 0.038, 0.025],
    'joint_axes': ['z', 'y', 'y', 'y'],
}

_LEFT_MIDDLE_CHAIN = {
    'base_pos': np.array([0.0, 0.0, -0.002]),
    'base_quat_angle': 0.0,
    'link_offsets': [0.017, 0.0431, 0.038, 0.025],
    'joint_axes': ['z', 'y', 'y', 'y'],
}

_LEFT_RING_CHAIN = {
    'base_pos': np.array([0.0005, 0.0425, -0.005]),
    'base_quat_angle': -0.0436194,
    'link_offsets': [0.017, 0.0431, 0.038, 0.025],
    'joint_axes': ['z', 'y', 'y', 'y'],
}

_LEFT_THUMB_CHAIN = {
    'base_pos': np.array([0.002, -0.027, -0.06342]),
    'link_offsets': [0.06874, 0.0, 0.038, 0.025],
    'joint_axes': ['x', 'z', 'y', 'y'],
}


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    if axis == 'x':
        return _rot_x(angle)
    elif axis == '-x':
        return _rot_x(-angle)
    elif axis == 'y':
        return _rot_y(angle)
    elif axis == 'z':
        return _rot_z(angle)
    raise ValueError(f"Unknown axis: {axis}")


def _finger_fk(joint_angles, chain_params):
    """Compute fingertip position given 4 joint angles and chain parameters.

    Returns the 3D position of the fingertip in the hand frame.
    """
    pos = chain_params['base_pos'].copy()

    # Base rotation (small tilt for index/ring)
    if 'base_quat_angle' in chain_params:
        angle = chain_params['base_quat_angle']
        rot = _rot_x(angle)
    else:
        rot = np.eye(3)

    offsets = chain_params['link_offsets']
    axes = chain_params['joint_axes']

    for i in range(4):
        # Apply joint rotation
        rot = rot @ _axis_rotation(axes[i], joint_angles[i])
        # Advance along local Z by the link offset
        pos = pos + rot @ np.array([0, 0, offsets[i]])

    return pos


def _finger_fk_jacobian(joint_angles, chain_params, eps=1e-5):
    """Numerical Jacobian of fingertip position w.r.t. joint angles."""
    pos0 = _finger_fk(joint_angles, chain_params)
    J = np.zeros((3, 4))
    for i in range(4):
        dq = joint_angles.copy()
        dq[i] += eps
        pos1 = _finger_fk(dq, chain_params)
        J[:, i] = (pos1 - pos0) / eps
    return J


# AVP finger joint indices for fingertip positions
_AVP_TIP_INDICES = {
    'thumb': 4,
    'index': 9,
    'middle': 14,
    'ring': 19,
    'little': 24,
}

# Allegro chain params indexed by finger name (right hand)
_ALLEGRO_CHAINS = {
    'index': _INDEX_CHAIN,
    'middle': _MIDDLE_CHAIN,
    'ring': _RING_CHAIN,
    'thumb': _THUMB_CHAIN,
}

# Left-hand chain params
_LEFT_ALLEGRO_CHAINS = {
    'index': _LEFT_INDEX_CHAIN,
    'middle': _LEFT_MIDDLE_CHAIN,
    'ring': _LEFT_RING_CHAIN,
    'thumb': _LEFT_THUMB_CHAIN,
}

# Joint index slices for each finger in the 16-dim Allegro vector
_ALLEGRO_SLICES = {
    'index': slice(0, 4),
    'middle': slice(4, 8),
    'ring': slice(8, 12),
    'thumb': slice(12, 16),
}


class HandRetargeter:
    """Retargets AVP hand skeleton to Allegro hand joint angles.

    Maps 5 AVP fingers to 4 Allegro fingers:
    - AVP index -> Allegro index
    - AVP middle -> Allegro middle
    - AVP (ring + little average) -> Allegro ring
    - AVP thumb -> Allegro thumb
    """

    def __init__(self, hand_scale: float = 1.0, side: str = 'right'):
        """
        Args:
            hand_scale: Scale factor to match AVP hand size to Allegro. Values > 1.0
                        mean the human hand is larger than the Allegro.
            side: 'left' or 'right' — selects mirrored kinematic chains for left hand.
        """
        self._hand_scale = hand_scale
        self._side = side
        self._chains = _LEFT_ALLEGRO_CHAINS if side == 'left' else _ALLEGRO_CHAINS
        self._prev_q = np.zeros(16)  # warm start
        # Set initial to mid-range
        for i in range(16):
            lo, hi = ALLEGRO_JOINT_LIMITS[i]
            self._prev_q[i] = (lo + hi) / 2.0

        self._bounds = [(lo, hi) for lo, hi in ALLEGRO_JOINT_LIMITS]

    def retarget(self, avp_fingers: np.ndarray) -> np.ndarray:
        """Retarget AVP finger data to Allegro joint angles.

        Args:
            avp_fingers: (25, 4, 4) array of finger joint transforms in wrist frame.

        Returns:
            (16,) array of Allegro joint angles.
        """
        # Extract target fingertip positions from AVP data
        targets = self._extract_targets(avp_fingers)

        # Optimize each finger independently (4 joints each, fast)
        result = self._prev_q.copy()

        for finger_name in ['index', 'middle', 'ring', 'thumb']:
            target_pos = targets[finger_name]
            chain = self._chains[finger_name]
            s = _ALLEGRO_SLICES[finger_name]
            q0 = self._prev_q[s].copy()
            bounds = self._bounds[s.start:s.stop]

            q_opt = self._optimize_finger(q0, target_pos, chain, bounds)
            result[s] = q_opt

        self._prev_q = result
        return result

    def _extract_targets(self, avp_fingers: np.ndarray) -> dict:
        """Extract and scale target fingertip positions from AVP data."""
        targets = {}

        # AVP wrist is at origin of this frame, fingertips are the "tip" joints
        for finger in ['thumb', 'index', 'middle']:
            tip_idx = _AVP_TIP_INDICES[finger]
            tip_pos = avp_fingers[tip_idx, :3, 3]  # translation component
            targets[finger] = tip_pos / self._hand_scale

        # Ring = average of ring and little fingertips
        ring_tip = avp_fingers[_AVP_TIP_INDICES['ring'], :3, 3]
        little_tip = avp_fingers[_AVP_TIP_INDICES['little'], :3, 3]
        targets['ring'] = (ring_tip + little_tip) / (2.0 * self._hand_scale)

        return targets

    def _optimize_finger(self, q0, target_pos, chain, bounds):
        """Optimize 4 joint angles to match a target fingertip position."""

        def cost(q):
            pos = _finger_fk(q, chain)
            err = pos - target_pos
            return 0.5 * np.sum(err ** 2)

        def grad(q):
            pos = _finger_fk(q, chain)
            err = pos - target_pos
            J = _finger_fk_jacobian(q, chain)
            return J.T @ err

        result = minimize(
            cost, q0, jac=grad, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20, 'ftol': 1e-10}
        )
        return result.x

    def retarget_from_fingertips(self, fingertip_positions: dict) -> np.ndarray:
        """Retarget from direct fingertip positions (in hand-local frame).

        Args:
            fingertip_positions: dict with 'thumb', 'index', 'middle', 'ring'
                each mapping to a (3,) position in the Allegro hand frame.

        Returns:
            (16,) array of Allegro joint angles.
        """
        result = self._prev_q.copy()

        for finger_name in ['index', 'middle', 'ring', 'thumb']:
            target_pos = fingertip_positions[finger_name] / self._hand_scale
            chain = self._chains[finger_name]
            s = _ALLEGRO_SLICES[finger_name]
            q0 = self._prev_q[s].copy()
            bounds = self._bounds[s.start:s.stop]

            q_opt = self._optimize_finger(q0, target_pos, chain, bounds)
            result[s] = q_opt

        self._prev_q = result
        return result
