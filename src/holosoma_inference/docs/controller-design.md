# Controller API — Design

## Problem

`BasePolicy` owns the SDK interface, input providers, run loop, FSM flags
(`use_policy_action`, `get_ready_state`, `_stiff_hold_active`), joint
interpolation, *and* the ONNX policy. Result: `_shared_hardware_source` hack for
dual-mode, duplicated run loop in `DualModePolicy`, and no clean place to add
"damping mode" or a startup FSM.

## Split

```
Controller           ← orchestrator: hardware, FSM, run loop, command dispatch
  └─ BasePolicy      ← pure obs→action: ONNX, history buffers, obs construction
```

`BasePolicy` no longer knows about `interface`, `rate`, or `run()`.
`Controller` is what `run_policy.py` instantiates and what `holosoma_service`
will run as a daemon.

## Controller states

```python
class ControllerState(Enum):
    IDLE         # no commands sent (motors slack)
    INIT         # interpolating to default/start pose over N ticks
    DAMP         # holds last observed q with low kp/kd — survives teleop release
    STIFF_HOLD   # holds configured pose with high kp/kd (WBT startup)
    RUN_POLICY   # delegates q_target to active BasePolicy
```

Transitions are command-driven (`StateCommand.START`, `INIT`, `STOP`,
`DAMP`, `KILL`, plus policy-specific). Each state is a pure function
`(robot_state) -> (q, kp_override, kd_override)`.

## Controller API

```python
class Controller:
    def __init__(
        self,
        config: InferenceConfig,
        policy: BasePolicy,                  # active policy (swappable)
        interface: RobotInterface,           # SDK
        velocity_input: VelCmdProvider,
        command_provider: StateCommandProvider,
    ): ...

    # Mutators called by command dispatch or external (service RPC, FSM):
    def set_state(self, state: ControllerState) -> None: ...
    def set_policy(self, policy: BasePolicy) -> None: ...   # dual-mode swap

    # Per-tick step — called by run() or by an external scheduler:
    def step(self) -> None: ...

    def run(self) -> None:                   # the only loop in the codebase
        for _ in itertools.count():
            self._poll_inputs()
            self.step()
            self.rate.sleep()
```

## `step()` — the orchestration

```python
def step(self) -> None:
    state_data = self.interface.get_low_state()

    if self.state is ControllerState.IDLE:
        return                                # do not publish

    elif self.state is ControllerState.INIT:
        q, kp, kd = self._interp_to(self.default_dof_angles, state_data)
        if self._init_done(): self.state = ControllerState.RUN_POLICY

    elif self.state is ControllerState.DAMP:
        if self._damp_q is None:
            self._damp_q = state_data[:, 7:7 + self.num_dofs].copy()
        q, kp, kd = self._damp_q, self.damp_kp, self.damp_kd

    elif self.state is ControllerState.STIFF_HOLD:
        q, kp, kd = self.policy.stiff_hold_target()      # policy-provided

    elif self.state is ControllerState.RUN_POLICY:
        action = self.policy.act(state_data)             # ← the only call into policy
        q = action + self.default_dof_angles
        kp, kd = None, None

    self.interface.send_low_command(q + self.joint_offsets, ..., kp, kd)
```

`_interp_to` and `_damp_q` live on `Controller`; they are *not* policy concerns.

## `BasePolicy` shrinks to

```python
class BasePolicy:
    def __init__(self, config, robot_config): ...        # no SDK, no inputs

    def act(self, robot_state_data) -> np.ndarray:       # was rl_inference
        obs = self.prepare_obs_for_rl(robot_state_data)
        return self.policy_fn(obs) * self.action_scale

    def stiff_hold_target(self) -> tuple[q, kp, kd]:     # default: raises
        raise NotImplementedError

    def on_velocity(self, vc: VelCmd) -> None: ...       # was _apply_velocity
    def on_command(self, cmd) -> ControllerState | None: # policy-specific
        # e.g. LocomotionPolicy handles STAND_TOGGLE here, returns None
        # WBT returns ControllerState.STIFF_HOLD on stop
        ...
```

## Damping mode wires up trivially

```python
# In _dispatch_command:
elif cmd == StateCommand.DAMP:
    self.controller.set_state(ControllerState.DAMP)
```

`holosoma_service` daemon: construct `Controller` once, default to `DAMP` on
startup, accept RPC/IPC to flip to `RUN_POLICY` when `teleop_app` connects, flip
back to `DAMP` on disconnect. No motors-go-slack on handle release.

## Dual-mode collapses to policy swap

```python
class DualModePolicy:
    def __init__(self, controller, primary, secondary):
        self.controller = controller
        self.policies = {"primary": primary, "secondary": secondary}
        self.active = "primary"

    def switch(self):
        self.active = "secondary" if self.active == "primary" else "primary"
        self.controller.set_policy(self.policies[self.active])
```

No second run loop. No `_shared_hardware_source` guard.

## Migration order

1. Add `Controller` + `ControllerState`; keep `BasePolicy.run()` as thin wrapper
   that constructs a Controller. No behavior change.
2. Move `interface`, `rate`, input providers, latency tracker, joint offsets,
   `policy_action()` body, and command dispatch into `Controller`.
3. Replace `BasePolicy.run()` with `Controller.run()` at call sites
   (`run_policy.py`, `DualModePolicy`).
4. Add `DAMP` state + `StateCommand.DAMP` mapping.
5. Delete `_shared_hardware_source` plumbing; rewrite `DualModePolicy` as swap.
6. (Future) Split `holosoma_service` (daemon w/ Controller) from `teleop_app`.

## Out of scope

- FSM transition graph beyond the 5 states above (Adam's "string utilities
  together" sequencing).
- Replacing tyro config plumbing.
- `holosoma_service` IPC protocol.
