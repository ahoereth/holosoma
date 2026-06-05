# Exoskeleton teleop service

Drives a Unitree G1 as a stop-gap tracker: arm joint targets → `rt/arm_sdk`,
base velocity → `LocoClient`. Subscribes to `ExoskeletonCmd` on
`/holosoma/tracker_command`. Runs on the Jetson, inside the holosoma-extensions
docker container.

## Robot state (before step 4)

The robot must be **standing and balanced** before launching the service:

1. Power on, hang the G1 on the gantry / have a spotter.
2. Joystick: **L2+Up** → stand.
3. Joystick: **R2+A** → running mode (legs under the loco controller, ready for walk).

The service then drives loco into **FSM-501** (conventional walk, arms decoupled
from legs) and runs the arm init trajectory. If the robot is sitting/damped when
you start, `loco.start()` won't bring it up correctly — get it standing first.

## Run

```bash
# 1. host: get a shell in the container
cd ~/projects/FAR-pi/holosoma_extensions && bash docker/run.sh

# 2. container: point CycloneDDS at the G1 interface (eth0)
export CYCLONEDDS_URI='<?xml version="1.0"?><CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="eth0"/></Interfaces></General></Domain></CycloneDDS>'

# 3. container: smoke-test the listener (builds the ROS msgs on first run, no robot motion)
python3 -m holosoma_inference.teleop.holosoma_teleop_listener_node

# 4. container: run the service (robot must be standing — see "Robot state" above)
python3 -m holosoma_inference.run_service               # arms + loco
python3 -m holosoma_inference.run_service --no-arms     # loco only
python3 -m holosoma_inference.run_service --no-loco     # arms only

# 5. container, SECOND shell: drive base velocity from the keyboard (SSH-safe)
python3 -m holosoma_inference.teleop.wasd_controller_node
#   w/s forward/back · a/d left/right · q/e yaw · space stop · Ctrl-C quit
```

## Notes

- First import builds `holosoma_teleop_msgs` via colcon into
  `/tmp/holosoma_teleop_ws` (needs `ROS_DISTRO` set; humble is fine). Wipe that
  dir to force a rebuild after editing a `.msg`.
- The service enters FSM-501 (arms-decoupled walk) before arm init — required,
  else `rt/arm_sdk` is ignored by the loco controller.
- Ctrl-C to stop (sends `StopMove`, closes the SDK clients).
