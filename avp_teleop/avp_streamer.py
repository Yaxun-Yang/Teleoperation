"""
Apple Vision Pro streaming interface using the avp_stream library.
Requires: pip install avp-stream
Requires: Apple Vision Pro running the Tracking Streamer visionOS app on the same network.
"""
import numpy as np
from avp_teleop.avp_interface import HandTrackingSource


class AVPStreamer(HandTrackingSource):
    """Streams hand/wrist tracking data from Apple Vision Pro via WebRTC.

    Usage:
        streamer = AVPStreamer(ip="192.168.1.100")
        while True:
            data = streamer.get_latest()
            # data['left_wrist'] -> (4,4) transform
            # data['left_fingers'] -> (25,4,4) finger joints
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
        raw = self._streamer.get_latest()
        if raw is None:
            return HandTrackingSource.make_identity_data()

        return {
            'head': np.array(raw.get('head', np.eye(4))),
            'left_wrist': np.array(raw.get('left_wrist', np.eye(4))),
            'right_wrist': np.array(raw.get('right_wrist', np.eye(4))),
            'left_fingers': np.array(raw.get('left_fingers', np.tile(np.eye(4), (25, 1, 1)))),
            'right_fingers': np.array(raw.get('right_fingers', np.tile(np.eye(4), (25, 1, 1)))),
        }

    def is_connected(self) -> bool:
        return self._connected
