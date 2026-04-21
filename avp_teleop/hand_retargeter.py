"""
Hand retargeting: DexPilot algorithm via the dex_retargeting library.

Maps AVP 25-joint hand skeleton to Allegro 16-joint angles using the
DexPilot optimization from dex_retargeting. This replaces the previous
scipy L-BFGS-B approach with a library-standard implementation that uses
inter-finger distance vectors as the optimization target.

Requires:
    pip install dex-retargeting torch --index-url https://download.pytorch.org/whl/cpu
    A dex-urdf clone with allegro_hand/ URDF files.
"""
import numpy as np
from pathlib import Path

from dex_retargeting.retargeting_config import RetargetingConfig


# AVP HandSkeleton tip joint indices (25 joints per hand)
TIP_IDX_THUMB = 4
TIP_IDX_INDEX = 9
TIP_IDX_MIDDLE = 14
TIP_IDX_RING = 19
TIP_IDX_LITTLE = 24


def _se3_inv(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform without np.linalg.inv."""
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def _shipped_dexpilot_yaml(hand_type: str) -> Path:
    """Return path to the DexPilot YAML shipped with dex_retargeting."""
    assert hand_type in ("right", "left"), hand_type
    import dex_retargeting as _dr
    pkg_dir = Path(_dr.__file__).parent
    yml = pkg_dir / "configs" / "teleop" / f"allegro_hand_{hand_type}_dexpilot.yml"
    if not yml.exists():
        raise FileNotFoundError(
            f"Shipped Allegro DexPilot config not found at {yml}. "
            f"Your dex_retargeting version may have moved it."
        )
    return yml


def _build_dexpilot_retargeter(hand_type: str):
    """Build a DexPilot retargeter for an Allegro hand."""
    yml = _shipped_dexpilot_yaml(hand_type)
    rc = RetargetingConfig.load_from_file(str(yml))
    return rc.build()


def _build_permutation(retargeter) -> np.ndarray:
    """Build permutation: retargeter output order -> MJCF sequential order (0..15).

    The DexPilot retargeter returns joints in URDF traversal order (e.g.
    joint_0.0, ..., joint_3.0, joint_12.0, ..., joint_15.0, joint_4.0, ...).
    The MJCF model expects joints 0..15 sequentially.

    Returns perm such that ``q_mjcf = q_retargeter[perm]``.
    """
    retar_names = list(retargeter.joint_names)

    def urdf_to_idx(name: str) -> int:
        return int(name.replace("joint_", "").replace(".0", ""))

    retar_idx_for_mjcf = {}
    for retar_i, name in enumerate(retar_names):
        mjcf_i = urdf_to_idx(name)
        retar_idx_for_mjcf[mjcf_i] = retar_i

    return np.array([retar_idx_for_mjcf[i] for i in range(16)], dtype=int)


class HandRetargeter:
    """DexPilot hand retargeter using the dex_retargeting library.

    Maps AVP 25-joint hand skeleton to Allegro 16-joint angles via the
    DexPilot optimization algorithm. Uses 10 inter-finger distance vectors
    (pinch, spread, palm-to-tip) as the optimization target.
    """

    def __init__(self, side: str = 'right', dex_urdf_dir: str = None):
        """
        Args:
            side: 'left' or 'right'
            dex_urdf_dir: Path containing ``allegro_hand/`` subdirectory with URDFs.
                         Must be set (here or globally) before the first call.
        """
        self._side = side

        if dex_urdf_dir is not None:
            RetargetingConfig.set_default_urdf_dir(str(dex_urdf_dir))

        self._retargeter = _build_dexpilot_retargeter(side)
        self._perm = _build_permutation(self._retargeter)

    def retarget(self, wrist_pose: np.ndarray, fingers_world: np.ndarray) -> np.ndarray:
        """Retarget AVP wrist + finger keypoints to Allegro joint angles.

        Args:
            wrist_pose: (4,4) wrist transform in any consistent frame.
            fingers_world: (25,4,4) finger keypoint transforms in the same frame.

        Returns:
            (16,) Allegro joint angles in MJCF order (joint 0..15).
        """
        vectors = self._build_retarget_input(wrist_pose, fingers_world)
        q_raw = self._retargeter.retarget(vectors)
        return np.asarray(q_raw, dtype=float)[self._perm]

    def _build_retarget_input(self, wrist_pose: np.ndarray,
                              fingers_world: np.ndarray) -> np.ndarray:
        """Build the (10, 3) vector array that DexPilot expects.

        The 10 vectors (in wrist frame) match the shipped DexPilot config's
        ``target_link_human_indices``::

            0: thumb  - index     (pinch)
            1: thumb  - middle    (pinch)
            2: thumb  - ring      (pinch)
            3: index  - middle    (spread)
            4: index  - ring      (spread)
            5: middle - ring      (spread)
            6: thumb  - wrist     (palm -> tip)
            7: index  - wrist     (palm -> tip)
            8: middle - wrist     (palm -> tip)
            9: ring   - wrist     (palm -> tip)
        """
        T_inv = _se3_inv(wrist_pose)

        def to_wrist(p_world: np.ndarray) -> np.ndarray:
            return (T_inv @ np.array([*p_world, 1.0]))[:3]

        thumb = to_wrist(fingers_world[TIP_IDX_THUMB][:3, 3])
        index = to_wrist(fingers_world[TIP_IDX_INDEX][:3, 3])
        middle = to_wrist(fingers_world[TIP_IDX_MIDDLE][:3, 3])
        ring = to_wrist(fingers_world[TIP_IDX_RING][:3, 3])
        wrist = np.zeros(3)

        return np.stack([
            thumb - index,
            thumb - middle,
            thumb - ring,
            index - middle,
            index - ring,
            middle - ring,
            thumb - wrist,
            index - wrist,
            middle - wrist,
            ring - wrist,
        ], axis=0)
