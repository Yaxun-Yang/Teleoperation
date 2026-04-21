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


def _get_dex_urdf_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'dex-urdf', 'robots', 'hands'
    )


def run_automated_test():
    """Quick automated test without GUI: verify model, DexPilot, mink IK."""
    import mink
    from dex_retargeting.retargeting_config import RetargetingConfig
    from avp_teleop.hand_retargeter import HandRetargeter

    model_path = _get_model_path()
    dex_urdf_dir = _get_dex_urdf_dir()
    print(f"=== Automated Test ===")
    print(f"Loading model: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}, nmocap={model.nmocap}")
    print(f"  Model loaded OK")

    # ---- Test DexPilot hand retargeting ----
    print("\n--- DexPilot Hand Retargeting ---")
    RetargetingConfig.set_default_urdf_dir(dex_urdf_dir)

    for side in ['right', 'left']:
        retargeter = HandRetargeter(side=side, dex_urdf_dir=dex_urdf_dir)

        # Synthetic wrist + fingers in MuJoCo frame
        wrist = np.eye(4)
        wrist[:3, 3] = [0.3, 0.0, 0.5]
        fingers = np.tile(np.eye(4), (25, 1, 1))
        # Place fingertips at plausible offsets from wrist
        fingers[4, :3, 3] = wrist[:3, 3] + [0.02, -0.03, 0.08]   # thumb
        fingers[9, :3, 3] = wrist[:3, 3] + [0.06, -0.02, 0.10]   # index
        fingers[14, :3, 3] = wrist[:3, 3] + [0.07, 0.00, 0.10]   # middle
        fingers[19, :3, 3] = wrist[:3, 3] + [0.06, 0.02, 0.09]   # ring
        fingers[24, :3, 3] = wrist[:3, 3] + [0.04, 0.04, 0.07]   # little

        q = retargeter.retarget(wrist, fingers)
        print(f"  {side} hand: {np.round(q, 3)}")
        assert q.shape == (16,), f"Expected (16,), got {q.shape}"
        assert np.all(np.isfinite(q)), f"Non-finite values in {side} hand retarget"
    print("  DexPilot retargeting: OK")

    # ---- Test mink IK ----
    print("\n--- mink Bimanual IK ---")
    configuration = mink.Configuration(model)
    q_init = data.qpos.copy()
    configuration.update(q_init)

    # Frame tasks for both EE sites
    left_task = mink.FrameTask(
        frame_name='mj_left_ee_site', frame_type='site',
        position_cost=1.0, orientation_cost=0.3, lm_damping=5.0)
    right_task = mink.FrameTask(
        frame_name='mj_right_ee_site', frame_type='site',
        position_cost=1.0, orientation_cost=0.3, lm_damping=5.0)
    posture = mink.PostureTask(model, cost=5e-2)
    posture.set_target_from_configuration(configuration)

    tasks = [left_task, right_task, posture]

    vel_limits = {}
    for side in ('left', 'right'):
        for i in range(1, 8):
            vel_limits[f'mj_{side}_joint{i}'] = 1.0
    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model, vel_limits),
    ]

    # Set a target slightly offset from current EE
    left_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'mj_left_ee_site')
    right_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'mj_right_ee_site')

    mujoco.mj_forward(model, data)
    T_left = np.eye(4)
    T_left[:3, :3] = data.site_xmat[left_site_id].reshape(3, 3)
    T_left[:3, 3] = data.site_xpos[left_site_id] + [0.05, 0.0, -0.05]
    T_right = np.eye(4)
    T_right[:3, :3] = data.site_xmat[right_site_id].reshape(3, 3)
    T_right[:3, 3] = data.site_xpos[right_site_id] + [-0.05, 0.0, -0.05]

    left_task.set_target(mink.SE3.from_matrix(T_left))
    right_task.set_target(mink.SE3.from_matrix(T_right))

    dt = 0.01
    for _ in range(100):
        vel = mink.solve_ik(configuration, tasks, dt,
                            solver="daqp", damping=5.0, limits=limits)
        configuration.integrate_inplace(vel, dt)

    # Verify arms moved
    left_adr = [model.joint(f'mj_left_joint{i}').qposadr[0] for i in range(1, 8)]
    right_adr = [model.joint(f'mj_right_joint{i}').qposadr[0] for i in range(1, 8)]
    q_left = configuration.q[left_adr]
    q_right = configuration.q[right_adr]
    print(f"  Left arm after IK:  {np.round(q_left, 3)}")
    print(f"  Right arm after IK: {np.round(q_right, 3)}")
    left_diff = np.linalg.norm(q_left - q_init[left_adr])
    right_diff = np.linalg.norm(q_right - q_init[right_adr])
    print(f"  Left arm moved: {left_diff:.4f} rad, Right arm moved: {right_diff:.4f} rad")
    assert left_diff > 0.001, f"Left arm didn't move enough: {left_diff}"
    assert right_diff > 0.001, f"Right arm didn't move enough: {right_diff}"
    print("  mink bimanual IK: OK")

    # ---- Test ManoMockHuman ----
    print("\n--- ManoMockHuman ---")
    from avp_teleop.mock_human import ManoMockHuman
    mano = ManoMockHuman(model, data)
    tracking = mano.get_latest()
    assert 'left_wrist' in tracking
    assert tracking['left_wrist'].shape == (4, 4)
    assert 'left_fingers' in tracking
    assert tracking['left_fingers'].shape == (25, 4, 4)
    print(f"  Left wrist pos: {np.round(tracking['left_wrist'][:3, 3], 4)}")
    print(f"  ManoMockHuman: OK")

    # ---- Test physics ----
    print("\n--- Physics Integration ---")
    for i in range(7):
        act_name = f'mj_left_act_pos{i+1}'
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        data.ctrl[act_id] = q_left[i]

    for _ in range(100):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    print(f"  Simulation stable: OK")

    print("\n=== All tests passed! ===")


def run_interactive(source_type, avp_ip=None):
    """Run interactive teleoperation with viewer."""
    from avp_teleop.teleop_controller import StandaloneTeleopController

    controller = StandaloneTeleopController(
        model_path=_get_model_path(),
        dex_urdf_dir=_get_dex_urdf_dir(),
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
