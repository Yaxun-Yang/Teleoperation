"""
Apple Vision Pro streaming interface using the avp_stream library.

Returns tracking data in AVP native frame (Y-up). The coordinate transform
to MuJoCo frame (Z-up) is handled by the teleop controller.

Requires:
    pip install avp-stream
    Apple Vision Pro running the Tracking Streamer visionOS app on the same network.
"""
import numpy as np
from avp_teleop.avp_interface import HandTrackingSource


class AVPStreamer(HandTrackingSource):
    """Streams hand/wrist tracking data from Apple Vision Pro via WebRTC.

    Usage:
        streamer = AVPStreamer(ip="192.168.1.100")
        while True:
            data = streamer.get_latest()
            # data['left_wrist']   -> (4,4) transform in AVP frame
            # data['left_fingers'] -> (25,4,4) finger joints in AVP frame
    """

    def __init__(self, ip_address: str, record: bool = False):
        try:
            from avp_stream import VisionProStreamer
        except ImportError:
            raise ImportError(
                "avp_stream not installed. Install with: pip install avp-stream\n"
                "Also requires Tracking Streamer app on Apple Vision Pro."
            )

        self._streamer = VisionProStreamer(ip=ip_address, record=record)
        self._connected = True
        print(f"AVPStreamer: Connecting to Vision Pro at {ip_address}...")

    def get_latest(self) -> dict:
        """Return the latest tracking data in AVP native frame (Y-up).

        Returns dict with:
            head: (4,4) head transform
            left_wrist / right_wrist: (4,4) wrist transforms
            left_fingers / right_fingers: (25,4,4) finger keypoint transforms
        """
        try:
            raw = self._streamer.latest
        except Exception:
            return HandTrackingSource.make_identity_data()

        if raw is None:
            return HandTrackingSource.make_identity_data()

        identity = HandTrackingSource.make_identity_data()

        def _get_pose(key: str) -> np.ndarray:
            """Extract a 4x4 pose from raw data (handles list-wrapped format)."""
            val = raw.get(key)
            if val is None:
                return identity.get(key, np.eye(4))
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 3 and arr.shape[0] >= 1:
                return arr[0]  # list-wrapped: take first element
            if arr.shape == (4, 4):
                return arr
            return identity.get(key, np.eye(4))

        def _get_fingers(key: str) -> np.ndarray:
            """Extract (25,4,4) finger keypoints from raw data."""
            val = raw.get(key)
            if val is None:
                return identity[key]
            arr = np.asarray(val, dtype=float)
            if arr.shape == (25, 4, 4):
                return arr
            return identity[key]

        return {
            'head': _get_pose('head'),
            'left_wrist': _get_pose('left_wrist'),
            'right_wrist': _get_pose('right_wrist'),
            'left_fingers': _get_fingers('left_fingers'),
            'right_fingers': _get_fingers('right_fingers'),
        }

    def is_connected(self) -> bool:
        return self._connected
