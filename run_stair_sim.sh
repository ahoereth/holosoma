#!/usr/bin/env bash
# MuJoCo sim for the depth-distillation stair policy: spawns the G1 with a
# forward-facing torso depth camera and publishes preprocessed depth to shared
# memory for run_stair_inference.sh to consume.
#
# Start this BEFORE the policy — the plugin creates the shared-memory block.
#
# See src/holosoma_inference/docs/workflows/sim-to-sim-depth-distillation.md
set -e

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO"

source scripts/source_mujoco_setup.sh

# MuJoCo opens a window whenever it can reach a display; without one it runs
# headless and the viewer-only gantry keys (7/8/9) are unavailable. Set DISPLAY
# yourself if your X server is not on :0.
if [ -z "${DISPLAY:-}" ]; then
    echo "[run_stair_sim] DISPLAY is unset — the MuJoCo window will not open and the" >&2
    echo "                gantry keys (7/8/9) will be unavailable. Export DISPLAY to fix." >&2
fi

# A stale block from a crashed run has the wrong size if the resolution changed.
rm -f /dev/shm/depth_img_shm 2>/dev/null || true

python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
    sensor.stair_front_depth:g1-stair-front-depth \
    plugin.depth:depth-shm \
    "$@"

# Notes:
#   - `sensor.<key>:<preset>` and `plugin.<key>:<preset>` are positional
#     declarations, not --flags. The key becomes the sensor/plugin name.
#   - The plugin's camera field must match the sensor key; the `depth-shm`
#     preset defaults to `stair_front_depth`, matching the key used above.
#   - Override the depth contract per-run if a checkpoint needs it, e.g.
#       ./run_stair_sim.sh --plugin.depth.near-clip 0.1 --plugin.depth.latency-frames 5
#   - Add a stair terrain once the .obj asset is available, e.g.
#       terrain:terrain-load-obj --terrain.terrain-term.obj-file-path <path>.obj
#     Without it the robot walks on a flat plane and the depth camera sees ground.
#
# MuJoCo window keys: 8 lower gantry (hold ~25 presses), 9 release, backspace reset.
