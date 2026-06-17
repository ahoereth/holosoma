# Holosoma teleop service

Two teleop backends, composed via `ros2 launch`. The API is the input ROS
messages; a backend turns them into robot motion.

```
CmdSMPLH ─▶ retargeter ─┐
                        ├─CmdDense─▶ holosoma_node (WBT) ─▶ G1   (holosoma policy)
external publisher ─────┘
CmdExoskeleton ──────────────────▶ unitree_split_controller ─▶ G1   (arm_sdk + loco)
```

The policy launch's `input_type` arg selects which `CmdDense` source is wired
in: `smplh` runs the retargeter; `dense` skips it for a direct publisher (see
[Launch arg: `input_type`](#launch-arg-input_type-policy-launch)).

The policy node never imports the policy extension by name: it resolves the
policy class from `config.task.policy_type` via the `holosoma.policies.by_type`
entry-point group and injects a live `DenseTargetSource` (subscribes `CmdDense`).
The `holosoma_wbt` policy is registered by a separate policy extension package,
which must be installed in the env alongside `holosoma` for the policy backend.

## Packages

ament colcon workspace of two packages:

- `holosoma_msgs` — message interfaces (`CmdSMPLH`, `CmdDense`, `CmdExoskeleton`,
  `Cmd3pt`, `Heartbeat`).
- `holosoma_service` — the ROS2 nodes (`holosoma_node`, `retargeter_node`,
  `unitree_split_controller`, `wasd_controller_node`) and launch files.

## Build & source

Build & source before launching; re-run after editing a `.msg`.

```bash
cd src/holosoma_inference/holosoma_inference_service
# CLEAN build — stale generated artifacts for old msg package names shadow the
# new ones. Nuke first whenever the .msg set changed:
rm -rf build install log
colcon build && source install/setup.bash
```

The policy backend additionally needs `holosoma` + a policy extension (the one
that registers your `holosoma.policies.by_type` entry point) importable in the
same environment. Verify the entry points resolve:

```bash
python -c "from holosoma_inference.compat import entry_points; \
  print([e.name for e in entry_points(group='holosoma.policies.by_type')])"
```

## Run — policy backend

```bash
# Whole-body WBT ONNX policy.
# preset defaults to g1-29dof-holosoma-wbt; input_type defaults to smplh.
ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
    urdf_path:=<fixed-base g1_29dof.urdf> model_path:=<model.onnx>
```

### Launch arg: `input_type` (policy launch)

`teleop_with_holosoma_policy.launch.py` takes `input_type` to select how the
`CmdDense` stream the policy consumes is produced:

| `input_type` | Nodes launched | `CmdDense` source | `urdf_path` |
|---|---|---|---|
| `smplh` (default) | retargeter + policy | `CmdSMPLH ─▶ retargeter ─▶ CmdDense` | **required** (retargeter IK model) |
| `dense` | policy only | external publisher feeds `CmdDense` directly | not used |

```bash
# SMPL-H teleop (retargeter on — the default):
ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
    urdf_path:=<fixed-base g1_29dof.urdf> model_path:=<model.onnx>

# Dense input (retargeter off — pair with any teleop already in dense
# 29-DOF convention, or a CmdDense replay publisher):
ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
    input_type:=dense model_path:=<model.onnx>
```

`input_type` is validated against `{smplh, dense}` at launch (a typo fails
fast).

## Run — split-body backend

```bash
# arm_sdk + LocoClient. Robot must be standing in FSM-501.
ros2 run holosoma_service unitree_split_controller --iface eth0
ros2 run holosoma_service unitree_split_controller --iface eth0 --no-arms   # loco only
```

A backend does nothing without an **input publisher** (your tracker / AVP / Pico
/ replay, or `ros2 run holosoma_service wasd_controller_node` for keyboard base
velocity).

## Input support (current)

| mode \ input                                      | `CmdSMPLH` (24-joint)        | `CmdDense` (29-DOF)          | `CmdExoskeleton` (arm q + twist) | `Cmd3pt` |
|---------------------------------------------------|------------------------------|------------------------------|----------------------------------|----------|
| **policy** (`teleop_with_holosoma_policy`)        | ✅ `input_type:=smplh` (retargeter) | ✅ `input_type:=dense` (direct) | ❌                               | ❌       |
| **split-body** (`teleop_with_unitree_split_body`) | ❌                           | ❌                           | ✅                               | ❌       |

## Gotchas

- **Two DDS graphs / two domains.** The unitree SDK link and the ROS2 graph each
  spin up a CycloneDDS participant. If both land on the same domain, the unitree
  binding aborts with `PreconditionNotMetError: Failed to create domain
  explicitly`. Keep the ROS graph (the node's `DenseTargetSource` + any
  `CmdDense` publisher) on a different `ROS_DOMAIN_ID` from the robot link.
- **`ros2 run` bypasses a conda env** — the installed console-script shebang is
  `#!/usr/bin/python3` (system Python). If your deps live in a conda env, run the
  node module directly with that env's Python via `python -m ...`. A rebuild is
  required after any `src/` edit — colcon copies `.py` into `build/`.
- **Policy URDF must be fixed-base** (no `<freejoint/>`) — the retargeter expects
  29 DOF; a freejoint URDF gives nq=36 and frames are rejected.
- **No input publisher = nothing moves.**
- `Cmd3pt` is defined but not yet consumed; Sonic policies not wired.
