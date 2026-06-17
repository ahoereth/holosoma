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
The `holosoma_wbt` entry point is registered by **wbt_wrappers** (FAR-pi), so the
policy backend requires `holosoma` + `wbt_wrappers_inference` installed in the env.

## Build & source

ament colcon workspace of two packages (`holosoma_msgs`, `holosoma_service`).
Build & source before launching; re-run after editing a `.msg`.

```bash
cd src/holosoma_inference/holosoma_inference_service
# CLEAN build — stale generated artifacts for old msg package names shadow the
# new ones. Nuke first whenever the .msg set changed:
rm -rf build install log
colcon build && source install/setup.bash
```

> **Known build quirk:** `catkin_pkg` is installed in the conda env, which
> perturbs ament's hook generation — `holosoma_msgs` ends up **missing its
> `ament_prefix_path` hook** (you'll see catkin hooks + a
> `CATKIN_INSTALL_INTO_PREFIX_ROOT` CMake warning). Python import and runtime
> DDS typesupport work fine, but `ros2` CLI tooling (`ros2 interface show`,
> `ros2 topic pub <type>`) can't see the package. Restore it after sourcing:
> ```bash
> export AMENT_PREFIX_PATH="$PWD/install/holosoma_msgs:$AMENT_PREFIX_PATH"
> ```

## Environment

Use the **hsmujoco_py312** env (Python 3.12 + ROS2 Jazzy + the unitree
CycloneDDS `LD_LIBRARY_PATH` fix baked in — needed so the unitree SDK binding
doesn't crash against ROS2 Jazzy's CycloneDDS):

```bash
source ~/projects/holosoma/scripts/source_mujoco_py312_setup.sh
```

This env has `holosoma`, `holosoma_inference`, `wbt_wrappers_inference`,
`unitree_interface`, `rclpy`, `mujoco`, `onnxruntime`, and `pinocchio` all
importable together. Verify the policy + sdk entry points resolve:

```bash
python -c "from holosoma_inference.compat import entry_points; \
  print([e.name for e in entry_points(group='holosoma.policies.by_type')])"
# expect: ['bm_wbt', 'holosoma_maskedmimic', 'holosoma_wbt', 'tml_wbt']
```

> **`ros2 run` does NOT work** for these Python nodes: the installed
> console-script shebang is `#!/usr/bin/python3` (system Python, no `tyro`).
> Run the module directly with the conda Python via `python -m ...` (see below).

## Run — sim2sim (no robot)

Drive the WBT policy against MuJoCo with **zero code changes**: `holosoma`'s
`run_sim` starts MuJoCo with a `UnitreeSdk2Bridge` (enabled by default) that
publishes `LowState` / reads `LowCmd` over the *same* unitree DDS protocol as a
real G1. The policy node's `unitree` SDK interface connects to it as if it were
hardware.

**Two DDS graphs, two domains** — the single most important thing to get right:

- The **unitree SDK ↔ sim** link is hardwired to **CycloneDDS domain 0** on
  interface `lo` (auto-detected). Both sides must match; don't change it.
- The **ROS2 graph** (the node's `DenseTargetSource` + any `CmdDense` publisher)
  runs on **`ROS_DOMAIN_ID=1`** so rclpy doesn't try to grab domain 0 too —
  otherwise the unitree binding aborts with
  `PreconditionNotMetError: Failed to create domain explicitly`.

**Ordered startup is required** (CycloneDDS multicast is disabled on `lo`, so
late-joining participants aren't reliably discovered). Launch **sim → node →
publisher**, in that order, and don't restart the node mid-run — a WBT tracker
can't stand back up if the robot falls while it's disconnected.

```bash
# --- Terminal 1: simulator (MuJoCo + unitree DDS bridge) ---
source ~/projects/holosoma/scripts/source_mujoco_py312_setup.sh
export ROS_DOMAIN_ID=0                       # robot link domain (matches bridge)
python -m wbt_wrappers_inference.run_sim simulator:mujoco
# A viewer opens; logs "LowCommandWriter - No motor command available!" until a
# policy connects (this is the bridge polling DDS for LowCmd — expected).

# --- Terminal 2: policy node ---
source ~/projects/holosoma/scripts/source_mujoco_py312_setup.sh
export ROS_DOMAIN_ID=1                        # ROS graph domain (NOT 0)
SVC=~/projects/holosoma/src/holosoma_inference/holosoma_inference_service
source $SVC/install/setup.bash
export AMENT_PREFIX_PATH="$SVC/install/holosoma_msgs:$AMENT_PREFIX_PATH"
DENSE=~/.cache/system_zero_leaderboard/weights/dense/model_29999.onnx
# ref-motion-path is required by the policy guard but IGNORED when a live
# DenseTargetSource is injected — pass any valid wbt_training NPZ.
REFM=~/projects/FAR-pi/holosoma_extensions/src/extensions/wbt_wrappers/wbt_training/motions/idle_stand_1.npz
python -m holosoma_service.policy_control.holosoma_node \
    inference:g1-29dof-holosoma-wbt \
    --task.model-path "$DENSE" \
    --task.ref-motion-path "$REFM" \
    --task.state-input keyboard
# Logs "RL FPS: 50.x" once connected. The "Non-interactive mode" warning is
# benign — the policy auto-engages ONNX, no keypress needed.
#
# Why --task.state-input keyboard: the preset sets state_input="interface",
# which makes the node try to read the Unitree SDK wireless controller
# ("Using joystick" in the log) — there's no controller in sim2sim. Forcing
# keyboard avoids that. (Cosmetic for the dense flow: motion comes from CmdDense
# via the injected DenseTargetSource, not from this input channel.)

# --- Terminal 3: input publisher (drive the policy) ---
source ~/projects/holosoma/scripts/source_mujoco_py312_setup.sh
export ROS_DOMAIN_ID=1
source $SVC/install/setup.bash
export AMENT_PREFIX_PATH="$SVC/install/holosoma_msgs:$AMENT_PREFIX_PATH"
# Replay a reference-motion NPZ as a live CmdDense feed (full trajectory):
python $SVC/holosoma_service/scripts/publish_from_npz.py \
    ~/projects/FAR-pi/holosoma_extensions/src/extensions/wbt_wrappers/wbt_training/motions/dance1_subject2.npz \
    --loop
```

The G1 in the viewer should track the motion (all 29 joints, staying upright).

### Specifics used above

| Thing | Value |
|---|---|
| Env | `hsmujoco_py312` via `scripts/source_mujoco_py312_setup.sh` |
| Dense ONNX | `~/.cache/system_zero_leaderboard/weights/dense/model_29999.onnx` |
| Preset | `inference:g1-29dof-holosoma-wbt` (obs_mode `dense`, 628-D obs, 4-frame hist, 29 DOF) |
| Fixed-base URDF | `.../wbt_wrappers/wbt_wrappers_inference/urdf/g1_29dof.urdf` (no `<freejoint/>`, 29 DOF) — only needed if you set `--task.robot-urdf-path` |
| Motions | `.../wbt_wrappers/wbt_training/motions/*.npz` (joint order already matches SDK) |
| Robot link | iface `lo`, `ROS_DOMAIN_ID=0` |
| ROS graph | `ROS_DOMAIN_ID=1` |

> **These ONNX/NPZ paths are local, uncommitted artifacts** — they are not in
> this repo. The dense ONNX lives under `~/.cache/system_zero_leaderboard/weights/`
> (populated by the `holosoma_tracker_eval` pipeline; see that extension's
> `OPENS.md` re: weight provenance). The motion NPZs ship in FAR-pi under
> `wbt_wrappers/wbt_training/motions/`. A fresh machine won't have the ONNX —
> point `--task.model-path` at any holosoma-format WBT checkpoint with a `dense`
> obs mode. Other holosoma-`dense` checkpoints seen on this box:
> `weights/{dense_v3/model_34999.onnx, dense_global_vel/model_29999.onnx}`.
> Known-good motions for `publish_from_npz.py`: `dance1_subject2.npz`,
> `dance2_subject1.npz`, `walk_forward_loop.npz`, `idle_stand_1.npz`.

**Dense mode needs no mocap** — its obs (`motion_command`, `motion_ref_ori_b`,
`projected_gravity`, `base_ang_vel`, `dof_pos`, `dof_vel`, `actions`) all derive
from `LowState`. The `sim_mocap` UDP bridge / `MocapInterfaceWrapper` are only
required for `global` / `2pt` / `3pt` obs modes.

## Run — real robot

```bash
# Policy mode (whole-body WBT ONNX). Needs holosoma + wbt_wrappers installed.
# preset defaults to g1-29dof-holosoma-wbt; input_type defaults to smplh.
ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
    urdf_path:=<fixed-base g1_29dof.urdf> model_path:=<model.onnx>

# Split-body mode (arm_sdk + LocoClient). Robot must be standing in FSM-501.
ros2 run holosoma_service unitree_split_controller --iface eth0
ros2 run holosoma_service unitree_split_controller --iface eth0 --no-arms   # loco only
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

# Dense input (retargeter off — pair with publish_from_npz.py or any teleop
# already in dense 29-DOF convention):
ros2 launch holosoma_service teleop_with_holosoma_policy.launch.py \
    input_type:=dense model_path:=<model.onnx>
```

`input_type` is validated against `{smplh, dense}` at launch (a typo fails
fast). Note the sim2sim recipe above runs `holosoma_node` directly via
`python -m` rather than `ros2 launch`, which is the `dense` flow by hand (no
retargeter); the launch file is the same topology for the real robot.

A backend does nothing without an **input publisher** (your tracker / AVP / Pico
/ replay, `publish_from_npz.py`, or `ros2 run holosoma_service wasd_controller_node`
for keyboard base vel).

## Input support (current)

| mode \ input                                      | `CmdSMPLH` (24-joint)        | `CmdDense` (29-DOF)          | `CmdExoskeleton` (arm q + twist) | `Cmd3pt` |
|---------------------------------------------------|------------------------------|------------------------------|----------------------------------|----------|
| **policy** (`teleop_with_holosoma_policy`)        | ✅ `input_type:=smplh` (retargeter) | ✅ `input_type:=dense` (direct) | ❌                               | ❌       |
| **split-body** (`teleop_with_unitree_split_body`) | ❌                           | ❌                           | ✅                               | ❌       |

## Gotchas

- **Two DDS graphs / two domains** — see sim2sim section. The #1 thing that bites.
- **Ordered startup** (sim → node → publisher) on `lo`; don't restart the node mid-run.
- **`ros2 run` bypasses the conda env** — use `python -m`.
- **tyro preset needs the `inference:` prefix.**
- **Preset forces SDK joystick input** — `g1-29dof-holosoma-wbt` sets
  `state_input="interface"` (log: `Using joystick`), which reads the Unitree
  wireless controller that doesn't exist in sim. Add `--task.state-input keyboard`.
- **`holosoma_msgs` missing `ament_prefix_path` hook** — `export AMENT_PREFIX_PATH` workaround.
- **Policy URDF must be fixed-base** (no `<freejoint/>`) — retargeter expects 29 DOF;
  a freejoint URDF gives nq=36 and frames are rejected.
- **No input publisher = nothing moves.**
- `Cmd3pt` is defined but not yet consumed; Sonic policies not wired.
