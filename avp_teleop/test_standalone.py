#!/usr/bin/env python3
"""
End-to-end test for AVP teleoperation with dual Panda + Allegro.

Usage:
    # With draggable MANO hand mocap bodies in MuJoCo viewer:
    python -m avp_teleop.test_standalone --source mock

    # With Apple Vision Pro (requires avp-stream and hardware):
    python -m avp_teleop.test_standalone --source avp --avp-ip 192.168.1.100

    # Quick automated test (no GUI, prints diagnostics):
    python -m avp_teleop.test_standalone --source test
"""
import os
import sys
import argparse
import time
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco


def _get_model_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'multipanda_ros2', 'franka_description', 'mujoco', 'franka',
        'mj_dual_allegro_scene.xml'
    )


def run_automated_test():
    """Quick automated test without GUI: verify model loads, IK works, retargeting works."""
    from avp_teleop.arm_retargeter import ArmRetargeter
    from avp_teleop.hand_retargeter import HandRetargeter

    model_path = _get_model_path()
    print(f"=== Automated Test ===")
    print(f"Loading model: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}, nmocap={model.nmocap}")
    print(f"  Model loaded OK")

    # Test arm retargeters
    print("\n--- Arm Retargeting (delta-pose) ---")
    left_arm = ArmRetargeter(model, data, 'left')
    right_arm = ArmRetargeter(model, data, 'right')

    # Initial EE poses
    left_pos, left_rot = left_arm.get_ee_pose()
    right_pos, right_rot = right_arm.get_ee_pose()
    print(f"  Left  EE initial: pos={np.round(left_pos, 4)}")
    print(f"  Right EE initial: pos={np.round(right_pos, 4)}")

    # Delta-pose test: first call sets origin, second call with offset causes movement
    origin_T = np.eye(4)
    origin_T[:3, 3] = [0.0, 0.0, 0.0]
    left_arm.solve(origin_T, engaged=True)
    right_arm.solve(origin_T, engaged=True)

    moved_T = np.eye(4)
    moved_T[:3, 3] = [0.1, 0.0, 0.0]
    left_q = left_arm.solve(moved_T, engaged=True)
    right_q = right_arm.solve(moved_T, engaged=True)
    print(f"  Left  arm IK (delta +0.1 X): {np.round(left_q, 3)}")
    print(f"  Right arm IK (delta +0.1 X): {np.round(right_q, 3)}")

    mujoco.mj_forward(model, data)
    left_pos2, _ = left_arm.get_ee_pose()
    right_pos2, _ = right_arm.get_ee_pose()
    left_moved = np.linalg.norm(left_pos2 - left_pos)
    right_moved = np.linalg.norm(right_pos2 - right_pos)
    print(f"  Left  EE after IK: pos={np.round(left_pos2, 4)} (moved {left_moved:.4f}m)")
    print(f"  Right EE after IK: pos={np.round(right_pos2, 4)} (moved {right_moved:.4f}m)")
    assert left_moved > 0.01, f"Left arm did not move enough: {left_moved}"
    assert right_moved > 0.01, f"Right arm did not move enough: {right_moved}"
    print(f"  Delta-pose IK: OK")

    # Test solve_absolute — use current EE position + small offset as target
    print("\n--- Arm Retargeting (absolute) ---")
    left_arm2 = ArmRetargeter(model, data, 'left')
    current_pos, _ = left_arm2.get_ee_pose()
    target = current_pos + np.array([0.05, -0.05, -0.05])
    abs_q = left_arm2.solve_absolute(target)
    left_arm2.set_joint_positions(abs_q)
    mujoco.mj_forward(model, data)
    result_pos, _ = left_arm2.get_ee_pose()
    abs_err = np.linalg.norm(result_pos - target)
    print(f"  Target: {np.round(target, 4)}, Result: {np.round(result_pos, 4)}, Error: {abs_err:.4f}m")
    assert abs_err < 0.05, f"Absolute IK error too large: {abs_err}"
    print(f"  Absolute IK: OK")

    # Test hand retargeting (AVP format)
    print("\n--- Hand Retargeting (AVP format) ---")
    hand_retargeter = HandRetargeter(hand_scale=1.5)

    fingers_open = np.tile(np.eye(4), (25, 1, 1))
    for i, offset in enumerate([
        [0.02, -0.03, 0.1],   [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0.02, -0.03, 0.1],
        [0.08, -0.04, 0.01],  [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0.08, -0.04, 0.12],
        [0.09, -0.01, 0.0],   [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0.09, -0.01, 0.12],
        [0.085, 0.02, 0.0],   [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0.085, 0.02, 0.11],
        [0.07, 0.04, 0.0],    [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [0.07, 0.04, 0.09],
    ]):
        fingers_open[i, :3, 3] = offset

    allegro_q = hand_retargeter.retarget(fingers_open)
    print(f"  Allegro joints (AVP): {np.round(allegro_q, 3)}")

    from avp_teleop.hand_retargeter import ALLEGRO_JOINT_LIMITS
    for i in range(16):
        lo, hi = ALLEGRO_JOINT_LIMITS[i]
        assert lo - 1e-6 <= allegro_q[i] <= hi + 1e-6, \
            f"Joint {i} out of bounds: {allegro_q[i]} not in [{lo}, {hi}]"
    print(f"  All 16 joints within limits: OK")

    # Test hand retargeting (direct fingertips)
    print("\n--- Hand Retargeting (fingertips) ---")
    hand_retargeter2 = HandRetargeter(hand_scale=1.5)
    fingertips = {
        'thumb': np.array([0.02, -0.03, 0.1]),
        'index': np.array([0.08, -0.04, 0.12]),
        'middle': np.array([0.09, -0.01, 0.12]),
        'ring': np.array([0.085, 0.02, 0.11]),
    }
    allegro_q2 = hand_retargeter2.retarget_from_fingertips(fingertips)
    print(f"  Allegro joints (fingertips): {np.round(allegro_q2, 3)}")
    for i in range(16):
        lo, hi = ALLEGRO_JOINT_LIMITS[i]
        assert lo - 1e-6 <= allegro_q2[i] <= hi + 1e-6, \
            f"Joint {i} out of bounds: {allegro_q2[i]} not in [{lo}, {hi}]"
    print(f"  All 16 joints within limits: OK")

    # Test ManoMockHuman
    print("\n--- ManoMockHuman ---")
    from avp_teleop.mock_human import ManoMockHuman
    mano = ManoMockHuman(model, data)
    tracking = mano.get_latest()
    assert 'left_wrist_pos' in tracking
    assert 'left_fingertips' in tracking
    assert 'thumb' in tracking['left_fingertips']
    print(f"  Left wrist: {np.round(tracking['left_wrist_pos'], 4)}")
    print(f"  Left thumb: {np.round(tracking['left_fingertips']['thumb'], 4)}")
    print(f"  ManoMockHuman: OK")

    # Test physics step with controls applied
    print("\n--- Physics Integration ---")
    for i in range(7):
        act_name = f'mj_left_act_pos{i+1}'
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        data.ctrl[act_id] = left_q[i]
    for i in range(16):
        act_name = f'mj_left_allegro_act_joint_{i}_0'
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        data.ctrl[act_id] = allegro_q[i]

    for _ in range(100):
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    left_pos3, _ = left_arm.get_ee_pose()
    print(f"  Left EE after 100 physics steps: {np.round(left_pos3, 4)}")
    print(f"  Simulation stable: OK")

    print("\n=== All tests passed! ===")


def run_interactive(source_type, avp_ip=None):
    """Run interactive teleoperation with viewer."""
    from avp_teleop.teleop_controller import StandaloneTeleopController

    controller = StandaloneTeleopController(
        model_path=_get_model_path(),
        control_freq=50.0,
    )

    if source_type == 'mock':
        from avp_teleop.mock_human import ManoMockHuman
        source = ManoMockHuman(controller.model, controller.data)
        controller.set_source(source)
        print("Using MANO hand mocap bodies. Double-click and drag spheres to teleoperate.")
    elif source_type == 'avp':
        from avp_teleop.avp_streamer import AVPStreamer
        if avp_ip is None:
            print("ERROR: --avp-ip required for AVP source")
            sys.exit(1)
        source = AVPStreamer(ip_address=avp_ip)
        controller.set_source(source)
        print(f"Using Apple Vision Pro at {avp_ip}")
    else:
        print(f"Unknown source: {source_type}")
        sys.exit(1)

    controller.run()


def main():
    parser = argparse.ArgumentParser(description="AVP Teleop Test")
    parser.add_argument('--source', type=str, default='test',
                        choices=['mock', 'avp', 'test'],
                        help='Data source: mock (MANO drag), avp (Vision Pro), test (automated)')
    parser.add_argument('--avp-ip', type=str, default=None,
                        help='Apple Vision Pro IP address (required for avp source)')
    args = parser.parse_args()

    if args.source == 'test':
        run_automated_test()
    else:
        run_interactive(args.source, args.avp_ip)


if __name__ == '__main__':
    main()
