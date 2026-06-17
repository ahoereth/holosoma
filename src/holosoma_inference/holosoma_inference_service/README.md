# Holosoma teleop service

Two teleop backends, composed via `ros2 launch`. The API is the input ROS messages; a backend turns them into robot motion.

## Service API
Inputs (one of):
- `CmdSMPLH.msg`
- `CmdDense.msg`
- `CmdExoskeleton.msg`
- `Cmd3pt.msg` (not supported yet)

Outputs:

| Topic | Type | Description | Rate |
|---|---|---|---|
| `/holosoma/holosoma_executed_cmd` | `JointState.msg` | Executed joint command, always full 29-DoF | Policy OR split-controller rate (typically 50Hz)  |
| `/holosoma/heartbeat` | `Heartbeat.msg` | Liveness + status (`robot_connected`, `control_mode`, `status`). Policy `status` is `running`/`get_ready`/`stiff_hold`; split-body is `running`/`waiting_for_cmd`. | 5 Hz |

Note: for  `/holosoma/holosoma_executed_cmd`, the policy backend fills all 29 values. The split-body backend fills the 14 arm joints (indices 15–28) and zeros the rest.

Internal structure:
```bash
CmdSMPLH ─▶ retargeter ─┐
                        ├─CmdDense─▶ holosoma_node (WBT) ─▶ G1   (holosoma policy)
external publisher ─────┘

CmdExoskeleton ──────────────────▶ unitree_split_controller ─▶ G1   (arm_sdk + loco)
```




## Input support

| mode \ input                                      | `CmdSMPLH` (24-joint)        | `CmdDense` (29-DOF)          | `CmdExoskeleton` (arm q + twist) | `Cmd3pt` |
|---------------------------------------------------|------------------------------|------------------------------|----------------------------------|----------|
| **policy** (`teleop_with_holosoma_policy`)        | ✅ `input_type:=smplh` (retargeter) | ✅ `input_type:=dense` (direct) | ❌                               | ❌       |
| **split-body** (`teleop_with_unitree_split_body`) | ❌                           | ❌                           | ✅                               | ❌       |



## Build & source

Build & source before launching; re-run after editing a `.msg`.

```bash
cd src/holosoma_inference/holosoma_inference_service
rm -rf build install log # for a clean build
colcon build && source install/setup.bash
```

## Run (with onnx policy)

Run with Whole-body WBT ONNX policy + SMPL-H teleop (the default)
```bash
ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
    urdf_path:=<fixed-base g1_29dof.urdf> \
    input_type:=simplh \
    model_path:=<model.onnx>
```

Whole body policy with `CmdDense` input (retargeter off):
```bash
ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
    input_type:=dense \
    model_path:=<model.onnx>
```

## Run (unitree split-body backend)

```bash
# arm_sdk + LocoClient. Robot must be standing in FSM-501.
ros2 run holosoma_service unitree_split_controller --iface eth0
ros2 run holosoma_service unitree_split_controller --iface eth0 --no-arms   # loco only
```

A backend does nothing without an **input publisher**. For the policy backend, the simplest one is the bundled NPZ replay script, which streams a reference-motion NPZ onto `CmdDense` as a live feed (pair with `input_type:=dense`):

```bash
python holosoma_service/scripts/publish_from_npz.py <motion.npz> --loop
```

Other publishers: your tracker / AVP / Pico, or `ros2 run holosoma_service wasd_controller_node` for mocking `CmdExoskeleton.msg`.
