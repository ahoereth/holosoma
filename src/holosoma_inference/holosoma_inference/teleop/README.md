# Teleop

This package holds the **rclpy-free core** of teleop: `retargeting/smpl_retargeter.py`
(`SMPLRetargeter`), pure-Python SMPL → 24-joint mink IK → holosoma (URDF/Mujoco
29-DOF) target. It has no ROS dependency and is unit-testable on its own.

Everything ROS — message definitions, nodes, launch — lives in the colcon
workspace at `../../holosoma_inference_service/` and is documented here because
the core is consumed from there.

## Architecture

```
SmplhCmd ──▶ retargeter_node ──DenseTrackingCmd──▶ policy_node ──▶ robot
(external      (runs SMPLRetargeter)                (WBT policy +
 tracker)                                            DenseTargetSource)
```

- `retargeter_node` subscribes `SmplhCmd` on `/holosoma/smplh_command`, runs the
  `SMPLRetargeter`, and publishes `DenseTrackingCmd` on
  `/holosoma/dense_tracking_command`.
- `policy_node` builds a `WholeBodyTrackingPolicy`, injects a `DenseTargetSource`
  (subscribes that topic) as the policy's target source, and runs the policy loop.
- `Heartbeat` (`holosoma_state_msgs`) is a status topic.

## Workspace layout (`holosoma_inference_service/`)

- `holosoma_input_msgs/` — ament interface pkg. Msgs: `SmplhCmd`,
  `ExoskeletonCmd`, `ThreePointCmd`, `DenseTrackingCmd`.
- `holosoma_state_msgs/` — ament interface pkg. Msg: `Heartbeat`.
- `holosoma_service/` — ament_python pkg: nodes + launch. console_scripts:
  `retargeter_node`, `policy_node`, `teleop_listener_node`, `wasd_controller_node`.

## Build & source

The msgs are real colcon-built ament packages — there is no longer any
auto-build-on-import. You **must** build the workspace and source it before any
`from holosoma_input_msgs.msg import ...` resolves or any node runs.

```bash
cd /path/to/holosoma_inference/holosoma_inference_service
colcon build
source install/setup.bash
```

Rebuild after editing a `.msg` (and re-`source`).

## Run

Launch the full teleop → policy flow:

```bash
ros2 launch holosoma_service teleop_policy.launch.py \
    urdf_path:=<fixed-base g1_29dof.urdf> \
    model_path:=<model.onnx> \
    preset:=g1-29dof-holosoma-wbt          # default; rl_rate_hz:=50.0 default
```

Or run nodes individually:

```bash
ros2 run holosoma_service retargeter_node --urdf-path <g1_29dof.urdf> --rl-rate-hz 50
ros2 run holosoma_service policy_node g1-29dof-holosoma-wbt --task.model-path <model.onnx>
ros2 run holosoma_service teleop_listener_node     # smoke-test listener, no robot motion
ros2 run holosoma_service wasd_controller_node     # keyboard base-velocity teleop (SSH-safe)
```

A `SmplhCmd` publisher (your tracking source — AVP / Pico / replay) is external
to this workspace. Without one, the retargeter has nothing to retarget.

## Status / notes

- **Untested scaffolding.** Needs a `colcon build` on the Jetson; the
  teleop → policy flow has not been verified end-to-end on hardware.
- `URDF must be fixed-base` (no `<freejoint/>`): the retargeter loads it as-is
  and expects 29 DOF; a freejoint URDF gives nq=36 and frames are rejected.
- `wasd_controller_node` reads stdin in cbreak mode (works over SSH):
  `w/s` fwd/back · `a/d` left/right · `q/e` yaw · `space` stop · Ctrl-C quit.
  It publishes `ExoskeletonCmd` (twist only) on `/holosoma/tracker_command`.

## Controller path (punted)

An earlier "controller" path drove the G1 directly via `rt/arm_sdk` + `LocoClient`
(stand with L2+Up, running mode R2+A, then FSM-501 arms-decoupled walk before arm
init), launched by a now-deleted `run_service.py`. That path is **punted** — it is
not part of the teleop → policy flow above. The `ExoskeletonCmd` /
`teleop_listener_node` / `wasd_controller_node` pieces remain in the workspace but
no node currently drives the Unitree SDK from them.
