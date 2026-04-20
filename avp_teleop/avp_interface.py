"""
Abstract base class for hand tracking data sources.
Provides a unified interface for both Apple Vision Pro and mock human model inputs.
"""
from abc import ABC, abstractmethod
import numpy as np


class HandTrackingSource(ABC):
    """Abstract interface for hand/wrist tracking data.

    All implementations must return data in the AVP-compatible format:
    {
        'head': np.ndarray (4,4),           # head transform in world frame
        'left_wrist': np.ndarray (4,4),     # left wrist transform
        'right_wrist': np.ndarray (4,4),    # right wrist transform
        'left_fingers': np.ndarray (25,4,4),  # left hand joint transforms
        'right_fingers': np.ndarray (25,4,4), # right hand joint transforms
    }

    AVP finger joint layout (25 joints per hand):
        Thumb:  indices 0-4  (metacarpal, CMC, MCP, IP, tip)
        Index:  indices 5-9  (metacarpal, MCP, PIP, DIP, tip)
        Middle: indices 10-14
        Ring:   indices 15-19
        Little: indices 20-24
    """

    @abstractmethod
    def get_latest(self) -> dict:
        """Return the latest tracking data."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the tracking source is active."""
        pass

    @staticmethod
    def make_identity_data() -> dict:
        """Return a default identity tracking frame."""
        eye4 = np.eye(4)
        return {
            'head': eye4.copy(),
            'left_wrist': eye4.copy(),
            'right_wrist': eye4.copy(),
            'left_fingers': np.tile(eye4, (25, 1, 1)),
            'right_fingers': np.tile(eye4, (25, 1, 1)),
        }
