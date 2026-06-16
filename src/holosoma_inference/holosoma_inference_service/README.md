# Holosoma teleop service

Two teleop backends, composed via `ros2 launch`. The API is the input ROS
messages; a backend turns them into robot motion.

```
SmplhCmd ─▶ retargeter ─DenseTrackingCmd─▶ run_policy (WBT) ─▶ G1   (holosoma policy)
ExoskeletonCmd ───────────────────────────▶ unitree_split_controller ─▶ G1   (arm_sdk + loco)
```

## Build & source (once, in the container on the Jetson)

The service is a colcon workspace of ament packages (`holosoma_input_msgs`,
`holosoma_state_msgs`, `holosoma_service`). No auto-build-on-import — build &
source before launching (re-run after editing a `.msg`):

```bash
# from this directory (holosoma_inference_service/)
colcon build && source install/setup.bash
```

## Run

```bash
# Policy mode (whole-body WBT ONNX). Needs both holosoma + wbt_wrappers installed.
ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
    urdf_path:=<fixed-base g1_29dof.urdf> model_path:=<model.onnx>   # preset defaults to g1-29dof-holosoma-wbt

# Split-body mode (arm_sdk + LocoClient). Robot must be standing in FSM-501.
ros2 launch holosoma_service teleop_with_unitree_split_body.launch.py iface:=eth0

# ...or run the controller node directly (tyro CLI; --no-arms/--no-loco bring up
# one client at a time during robot bringup):
ros2 run holosoma_service unitree_split_controller --iface eth0
ros2 run holosoma_service unitree_split_controller --iface eth0 --no-arms   # loco only
```

A backend does nothing without an **input publisher** (your tracker / AVP / Pico
/ replay, or `ros2 run holosoma_service wasd_controller_node` for keyboard base vel).

## Input support (current)

| mode \ input                                      | `SmplhCmd` (24-joint) | `ExoskeletonCmd` (arm q + twist) | `ThreePointCmd` |
|---------------------------------------------------|-----------------------|---------------------------------|-----------------|
| **policy** (`teleop_with_holosoma_policy`)        | ✅ via retargeter     | ❌                              | ❌              |
| **split-body** (`teleop_with_unitree_split_body`) | ❌                    | ✅                              | ❌              |

## Gotchas

- **Untested** — needs a `colcon build` on the Jetson; not verified end-to-end.
- **Policy URDF must be fixed-base** (no `<freejoint/>`) — retargeter expects 29 DOF;
  a freejoint URDF gives nq=36 and frames are rejected.
- **No input publisher = nothing moves.**
- `ThreePointCmd` is defined but not yet consumed; Sonic policies not wired.
