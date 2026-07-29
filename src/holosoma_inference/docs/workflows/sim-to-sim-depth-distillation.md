# Sim-to-Sim Depth Distillation Workflow

> **See also:** [Inference & Deployment Guide](../../README.md) for all deployment options

This guide provides a complete workflow for running vision-based locomotion (depth
distillation) policies in MuJoCo simulation.

## Overview

Depth distillation policies see the terrain ahead, so they can place footholds on
stairs and rough ground rather than walking blind. A policy is a pair of ONNX models:

- **depth backbone** — encodes a depth image into a compact latent
- **student** — maps proprioception + a direction command + that latent to joint targets

The sim renders depth from a torso-mounted camera and publishes it to shared memory;
the policy process reads it each control tick. The same policy code runs on hardware
against an on-robot depth server publishing the same format.

## Prerequisites

- MuJoCo environment set up (`scripts/source_mujoco_setup.sh`)
- Holosoma inference environment set up (`scripts/source_inference_setup.sh`)
- A checkpoint **pair**: `<run>_depth_backbone.onnx` and `<run>_student.onnx`
- Keyboard for control

**Note:** Always use `--task.interface lo` (loopback) when inference and MuJoCo run on
the same machine.

**Note:** Start the simulator **before** the policy — the sim creates the shared-memory
block the policy attaches to.

---

## Unitree G1 (29-DOF)

### 1. Start MuJoCo Environment

In one terminal, launch the simulator with the depth camera and the shared-memory
publisher:

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    sensor.stair_front_depth:g1-stair-front-depth \
    plugin.depth:depth-shm
```

The robot will spawn in the simulator, hanging from a gantry. The log should show:

```
[DepthShmPlugin] created 'depth_img_shm' (20184 bytes) shape=(1, 1, 58, 87)
```

`sensor.<key>:<preset>` and `plugin.<key>:<preset>` are positional declarations, not
`--flags`. The plugin's `camera` field must name the sensor key; the `depth-shm` preset
defaults to `stair_front_depth`, matching the key above.

### 2. Launch the Policy

In another terminal, run the policy inference:

```bash
source scripts/source_inference_setup.sh
CKPT=/path/to/checkpoints/2026-07-03_05-20-32_rfh-rvol-grid-fh5.0-rv10.0
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-wbt-distillation \
    --task.interface lo \
    --task.model-path "['${CKPT}_depth_backbone.onnx','${CKPT}_student.onnx']" \
    --task.record-dir stair \
    --task.record-label stair
```

Confirm the policy attached to the sim's depth stream:

```
[DepthShmSensor] attached to 'depth_img_shm' shape=(1, 1, 58, 87)
```

Both model paths go in **one quoted Python list literal**. `TaskConfig.model_path` is
typed `str | list[str]` and the CLI resolves that union to `str`, so space-separated
paths are parsed as a single token and fail. Comma-separated also works.

### 3. Deploy the Robot

- In MuJoCo window, press `8` repeatedly (~25 times) to lower the gantry until the feet
  touch the ground
- In MuJoCo window, press `9` to remove the gantry

The policy holds a stiff standing pose until started, so the robot stays upright while
the gantry comes down.

### 4. Start the Policy

In policy terminal, press `]` to activate the policy.

### 5. Control the Robot

In policy terminal, use `w` `a` `s` `d` `q` `e` to steer and `=` to change speed mode.
Each key selects a heading outright — see below.

---

## Convenience Scripts

Both steps above are wrapped:

```bash
./run_stair_sim.sh          # terminal 1
./run_stair_inference.sh    # terminal 2
```

Extra arguments pass through to the underlying command. Point at a different
checkpoint pair with environment variables:

```bash
CKPT_DIR=/path/to/ckpts RUN=<run-name> ./run_stair_inference.sh
```

To exercise the control loop without the simulator (zero-filled depth frames):

```bash
./run_stair_inference.sh --task.depth-shm.no-required
```

---

## Policy Controls Reference

**Enter these commands in the policy terminal** (where you ran `run_policy.py`).

### General Controls

| Action | Keyboard |
|--------|----------|
| Start the policy | `]` |
| Damping mode (Kp=0, Kd>0) | `o` |
| Re-enter stiff hold | `i` |
| Toggle motion recording | `c` |

`o` enters a damping mode rather than zeroing all gains, so the robot yields but
resists free-fall. `i` eases back to the startup pose over 2 s.

### Direction Controls

| Action | Keyboard |
|--------|----------|
| Forward / backward | `w` / `s` |
| 45° left / right | `a` / `d` |
| 90° left / right | `q` / `e` |
| Stand | `z` |
| Cycle speed mode (LOW → HIGH → MADMAX) | `=` |

Unlike the blind locomotion policy, direction is a **discrete class**, not a continuous
velocity — that is how the command was represented during training. Each press
therefore selects an **absolute heading**: one tap responds immediately, and reversing
from forward to back takes a single press.

Joystick and ROS2 still deliver continuous velocities; those are quantized to the
nearest direction sector, so all input sources work.

### MuJoCo Controls Reference

**Enter these in the MuJoCo window** (not the policy terminal):

- `7` / `8`: Raise / lower the gantry
- `9`: Disable/remove the gantry
- `Backspace`: Reset simulation

---

## Recording Motion Clips

Press `c` while the policy is running to start recording, `c` again to stop and save.
Stopping the policy (`o`) or re-entering stiff hold (`i`) autosaves an in-flight
recording, so a clip is never silently lost.

Clips are written to
`recorded_motion/<record-dir>/<record-label>_duration<X.X>s_motion.npz`, **relative to
the working directory**, containing:

| Key | Shape | Contents |
|-----|-------|----------|
| `qpos` | `[N, 36]` float32 | root quaternion `[0:4]` (wxyz), root position `[4:7]`, absolute joint positions `[7:36]` in robot order |
| `vel_cmd` | `[N, 15]` float32 | the one-hot direction command per frame |
| `fps` | scalar int64 | control rate (50) |

Note `qpos` is **quaternion-first**, which differs from the position-first ordering used
by MuJoCo-format training clips. The clip duration is part of the filename, so two clips
of the same length overwrite each other (the policy logs a warning when it does).

---

## Configuration

### Depth preprocessing

The simulator does the resize, clip and normalize so the policy reads a tensor it can
hand straight to the backbone — this keeps the two sides from disagreeing about it.
The values must match what the checkpoint was trained with:

```bash
python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    sensor.stair_front_depth:g1-stair-front-depth plugin.depth:depth-shm \
    --plugin.depth.near-clip 0.1 \
    --plugin.depth.far-clip 2.0 \
    --plugin.depth.latency-frames 5
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--plugin.depth.near-clip` | `0.3` | Depth at or below this normalizes to `-0.5` |
| `--plugin.depth.far-clip` | `2.0` | Depth at or above this normalizes to `+0.5` |
| `--plugin.depth.latency-frames` | `0` | Publish a frame this many steps old, modeling real camera/transport delay (5 frames at 50 Hz = 100 ms) |
| `--plugin.depth.shm-name` | `depth_img_shm` | Shared-memory block name; must match `--task.depth-shm.name` |

The resize uses bicubic interpolation with antialiasing to match
`torchvision.transforms.Resize(..., BICUBIC)` as used in training. Bilinear resizing
shifts the depth statistics enough to degrade foothold placement.

### Camera resolution

If you change the render resolution, the resized output must still match the backbone's
input — the policy fails loud on a size mismatch rather than reading garbage:

```bash
--sensor.stair_front_depth.width 320 --sensor.stair_front_depth.height 180
```

### Stair terrain

The default terrain is a flat plane: this repo ships `MeshType.LOAD_OBJ` but no terrain
mesh assets. The depth camera sees ground and the policy runs, but for actual stairs
supply a mesh:

```bash
python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    sensor.stair_front_depth:g1-stair-front-depth plugin.depth:depth-shm \
    terrain:terrain-load-obj \
    --terrain.terrain-term.obj-file-path <path>/chained_stairs_15.obj
```

---

## How It Works

```
MuJoCo depth render (240x135, metric meters)
  └─ DepthShmPlugin: resize -> 58x87 (bicubic+antialias),
     clip [0.3, 2.0], normalize [-0.5, 0.5]
       └─ /dev/shm/depth_img_shm : float32 (1, 1, 58, 87), 50 Hz
            └─ DepthShmSensor (policy process)
                 └─ depth_backbone.onnx -> latent [1, 32]
                      └─ student.onnx: obs [1, 140] -> actions [1, 29]
```

Shared memory rather than a ROS2 topic because the consumer runs on the same host at
the control rate: no serialization, no broker, no per-frame allocation.

The student's 140-dim input is a **wire format**:

| Slice | Contents |
|-------|----------|
| `[0:3]` | `projected_gravity` — in the **torso** frame, not the pelvis |
| `[3:6]` | `base_ang_vel` |
| `[6:35]` | `dof_pos` (in **model** joint order) |
| `[35:64]` | `dof_vel` (model order) |
| `[64:93]` | previous `actions` (model order) |
| `[93:108]` | one-hot direction command |
| `[108:140]` | depth latent |

Three details there are load-bearing. Each would produce a plausible-looking but wrong
action rather than an error, so all three are pinned by tests:

- **Term order is declaration order, not alphabetical.** The rest of this package sorts
  observation term names; these checkpoints do not. The preset sets
  `sort_obs_terms=False`, so the order written in `obs_dict` *is* the layout —
  reordering that list silently changes the observation vector.
- **Joint order differs from the robot's.** The checkpoint's `joint_names` metadata is
  IsaacLab's interleaved ordering. Observations are permuted into model order and
  actions back into robot order. `last_policy_action` is deliberately kept in *model*
  order, because it feeds back as the `actions` observation on the next tick.
- **Gravity is observed in the anchor (torso) frame.** On hardware the IMU sits in the
  torso, so the base quaternion already is the torso's; in MuJoCo the floating base is
  the pelvis, so the waist yaw/roll/pitch rotations are chained on.

Control gains come from the **robot config**, not the student's ONNX metadata. The
exporter writes per-joint `joint_stiffness`/`joint_damping` that differ from the
deployed values by a few percent (hip_pitch 38.869 vs 40.179); the reference deployment
runs the config values, so the config takes priority.

---

## Tips and Troubleshooting

- **`Depth shared memory 'depth_img_shm' not found`**: the simulator is not running, or
  exited (it unlinks the block on close). Start the sim first, or pass
  `--task.depth-shm.no-required` to run with zero-filled depth.
- **`Existing shared memory ... is N bytes but this camera needs M`**: a stale block from
  a run at a different resolution. Remove `/dev/shm/depth_img_shm` and retry.
- **`requires exactly 2 model paths`**: the two paths collapsed into one token. Quote
  them as a single Python list literal (see step 2).
- **Keyboard does nothing**: keyboard input needs a TTY. Without one the policy logs
  `No TTY — keyboard input disabled` and auto-starts, so it runs but cannot be steered.
- **Reset anytime**: press `Backspace` in the MuJoCo window.
- **Interface**: always use `lo` (loopback) for sim-to-sim on the same machine.
