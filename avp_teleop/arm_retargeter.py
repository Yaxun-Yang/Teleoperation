"""
Arm retargeting: maps AVP wrist transforms to Panda arm joint angles.
Uses MuJoCo's built-in Jacobian for damped least-squares IK.
Implements delta-pose engagement pattern from teleop_panda.py.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def _mat_to_quat(mat):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z] (MuJoCo convention)."""
    r = Rotation.from_matrix(mat)
    q = r.as_quat()  # scipy returns [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])


def _quat_to_mat(quat):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = np.array([quat[1], quat[2], quat[3], quat[0]])  # scipy wants [x, y, z, w]
    return Rotation.from_quat(q).as_matrix()


def _quat_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    r = r1 * r2
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _quat_diff(q1, q2):
    """Compute rotation from q1 to q2: q_diff such that q2 = q1 * q_diff."""
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    r_diff = r1.inv() * r2
    q = r_diff.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _orientation_error(target_mat, current_mat):
    """Compute 3D orientation error vector between two rotation matrices."""
    r_err = Rotation.from_matrix(target_mat @ current_mat.T)
    return r_err.as_rotvec()


# AVP to MuJoCo coordinate frame transform
# AVP: right-handed, Y-up. MuJoCo: right-handed, Z-up.
# Mapping: mj_x = avp_x, mj_y = -avp_z, mj_z = avp_y
_AVP_TO_MJ_ROT = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0],
], dtype=float)


def avp_to_mujoco_transform(avp_T):
    """Convert a 4x4 transform from AVP frame to MuJoCo frame."""
    mj_T = np.eye(4)
    mj_T[:3, :3] = _AVP_TO_MJ_ROT @ avp_T[:3, :3] @ _AVP_TO_MJ_ROT.T
    mj_T[:3, 3] = _AVP_TO_MJ_ROT @ avp_T[:3, 3]
    return mj_T


class ArmRetargeter:
    """Retargets AVP wrist pose to Panda arm joint angles using MuJoCo IK.

    Uses delta-pose control: tracks relative motion from an engagement origin,
    not absolute positions.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        side: 'left' or 'right'
        pos_scale: scaling factor for position deltas
        rot_scale: scaling factor for rotation deltas
    """

    def __init__(self, model, data, side: str, pos_scale=1.0, rot_scale=1.0):
        self._model = model
        self._data = data
        self._side = side
        self._pos_scale = pos_scale
        self._rot_scale = rot_scale

        # Joint indices for this arm (7 joints)
        self._joint_names = [f'mj_{side}_joint{i}' for i in range(1, 8)]
        self._joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                           for n in self._joint_names]
        self._qpos_indices = [model.jnt_qposadr[jid] for jid in self._joint_ids]
        self._dof_indices = [model.jnt_dofadr[jid] for jid in self._joint_ids]

        # EE site
        self._ee_site_name = f'mj_{side}_ee_site'
        self._ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self._ee_site_name)

        # Joint limits
        self._q_min = np.array([model.jnt_range[jid][0] for jid in self._joint_ids])
        self._q_max = np.array([model.jnt_range[jid][1] for jid in self._joint_ids])

        # Delta-pose tracking state
        self._engaged = False
        self._avp_origin_pos = np.zeros(3)
        self._avp_origin_rot = np.eye(3)
        self._robot_origin_pos = np.zeros(3)
        self._robot_origin_rot = np.eye(3)

        # IK parameters
        self._damping = 0.05
        self._ik_steps = 20
        self._step_size = 0.5

        # Target pose (updated each frame)
        self._target_pos = None
        self._target_rot = None

        # Jump detection threshold
        self._max_pos_delta = 0.3
        self._max_rot_delta = 1.5

    def get_joint_positions(self) -> np.ndarray:
        """Get current 7-DOF joint positions."""
        return np.array([self._data.qpos[i] for i in self._qpos_indices])

    def set_joint_positions(self, q):
        """Set 7-DOF joint positions in the MuJoCo data."""
        for i, idx in enumerate(self._qpos_indices):
            self._data.qpos[idx] = q[i]

    def get_ee_pose(self):
        """Get current EE position and rotation matrix."""
        mujoco.mj_forward(self._model, self._data)
        pos = self._data.site_xpos[self._ee_site_id].copy()
        rot = self._data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
        return pos, rot

    def engage(self):
        """Start tracking: save current AVP and robot origins."""
        self._engaged = True

    def disengage(self):
        """Stop tracking: will reset origins on next engage."""
        self._engaged = False

    def solve(self, avp_wrist_T: np.ndarray, engaged: bool = True) -> np.ndarray:
        """Compute target joint angles from an AVP wrist transform.

        Args:
            avp_wrist_T: (4,4) wrist transform from AVP in AVP frame
            engaged: whether the user is actively controlling (True = track motion)

        Returns:
            (7,) array of target joint positions for this arm
        """
        # Convert AVP frame to MuJoCo frame
        mj_T = avp_to_mujoco_transform(avp_wrist_T)
        avp_pos = mj_T[:3, 3]
        avp_rot = mj_T[:3, :3]

        if not engaged:
            # Not engaged: just update origins
            self._avp_origin_pos = avp_pos.copy()
            self._avp_origin_rot = avp_rot.copy()
            ee_pos, ee_rot = self.get_ee_pose()
            self._robot_origin_pos = ee_pos.copy()
            self._robot_origin_rot = ee_rot.copy()
            return self.get_joint_positions()

        if not self._engaged:
            # First frame of engagement: set origins
            self._avp_origin_pos = avp_pos.copy()
            self._avp_origin_rot = avp_rot.copy()
            ee_pos, ee_rot = self.get_ee_pose()
            self._robot_origin_pos = ee_pos.copy()
            self._robot_origin_rot = ee_rot.copy()
            self._engaged = True

        # Compute delta from AVP origin
        d_pos = (avp_pos - self._avp_origin_pos) * self._pos_scale
        d_rot_mat = self._avp_origin_rot.T @ avp_rot

        # Jump detection
        if np.linalg.norm(d_pos) > self._max_pos_delta:
            # Re-center on current position
            self._avp_origin_pos = avp_pos.copy()
            ee_pos, _ = self.get_ee_pose()
            self._robot_origin_pos = ee_pos.copy()
            return self.get_joint_positions()

        d_rot_angle = np.linalg.norm(Rotation.from_matrix(d_rot_mat).as_rotvec())
        if d_rot_angle > self._max_rot_delta:
            self._avp_origin_rot = avp_rot.copy()
            _, ee_rot = self.get_ee_pose()
            self._robot_origin_rot = ee_rot.copy()
            return self.get_joint_positions()

        # Compute target in robot frame
        self._target_pos = self._robot_origin_pos + d_pos
        self._target_rot = self._robot_origin_rot @ d_rot_mat

        # Run IK to find joint angles
        return self._solve_ik()

    def solve_absolute(self, target_pos: np.ndarray, target_rot: np.ndarray = None) -> np.ndarray:
        """Position-only IK toward a world-frame target.

        No delta-pose, no AVP coordinate transform — direct world-frame target.
        Saves/restores qpos so physics isn't bypassed.

        Args:
            target_pos: (3,) target position in MuJoCo world frame.
            target_rot: (3,3) optional target rotation. If None, position-only IK.

        Returns:
            (7,) array of target joint positions.
        """
        self._target_pos = target_pos.copy()

        # Save full qpos so IK doesn't pollute the physics state
        qpos_save = self._data.qpos.copy()

        if target_rot is not None:
            self._target_rot = target_rot.copy()
            q = self._solve_ik(ik_steps=50, step_size=0.5)
        else:
            q = self._solve_ik_position_only(ik_steps=50, step_size=0.5)

        # Restore qpos — physics will drive the arm via actuators
        self._data.qpos[:] = qpos_save
        mujoco.mj_forward(self._model, self._data)

        return q

    def _solve_ik_position_only(self, ik_steps=50, step_size=0.5) -> np.ndarray:
        """Position-only damped least-squares IK (3D, not 6D)."""
        q = self.get_joint_positions()
        damping = self._damping

        for _ in range(ik_steps):
            self.set_joint_positions(q)
            mujoco.mj_forward(self._model, self._data)
            ee_pos = self._data.site_xpos[self._ee_site_id].copy()

            pos_err = self._target_pos - ee_pos
            if np.linalg.norm(pos_err) < 1e-4:
                break

            jacp = np.zeros((3, self._model.nv))
            body_id = self._model.site_bodyid[self._ee_site_id]
            point = self._data.site_xpos[self._ee_site_id]
            mujoco.mj_jac(self._model, self._data, jacp, None, point, body_id)

            J = jacp[:, self._dof_indices]
            JJT = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, pos_err)

            q = q + step_size * dq
            q = np.clip(q, self._q_min, self._q_max)

        return q

    def _solve_ik(self, ik_steps=None, step_size=None) -> np.ndarray:
        """Damped least-squares IK to reach target_pos/target_rot (6D)."""
        if ik_steps is None:
            ik_steps = self._ik_steps
        if step_size is None:
            step_size = self._step_size

        q = self.get_joint_positions()

        for _ in range(ik_steps):
            # Forward kinematics
            self.set_joint_positions(q)
            mujoco.mj_forward(self._model, self._data)
            ee_pos = self._data.site_xpos[self._ee_site_id].copy()
            ee_rot = self._data.site_xmat[self._ee_site_id].reshape(3, 3).copy()

            # Position error
            pos_err = self._target_pos - ee_pos

            # Orientation error
            rot_err = _orientation_error(self._target_rot, ee_rot)

            # Combined 6D error
            err = np.concatenate([pos_err, rot_err * self._rot_scale])

            if np.linalg.norm(err) < 1e-4:
                break

            # Compute Jacobian
            jacp = np.zeros((3, self._model.nv))
            jacr = np.zeros((3, self._model.nv))

            body_id = self._model.site_bodyid[self._ee_site_id]
            point = self._data.site_xpos[self._ee_site_id]
            mujoco.mj_jac(self._model, self._data, jacp, jacr, point, body_id)

            # Extract columns for our joints only
            J_pos = jacp[:, self._dof_indices]
            J_rot = jacr[:, self._dof_indices]
            J = np.vstack([J_pos, J_rot])

            # Damped least-squares
            JJT = J @ J.T + self._damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, err)

            # Update joint positions
            q = q + step_size * dq
            q = np.clip(q, self._q_min, self._q_max)

        return q
