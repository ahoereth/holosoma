# Controller refactor — Design

## Status

| Step | What | Status |
|---|---|---|
| 0 | Sim2sim test harness | done (`846c299`) |
| 1 | Extract `Controller` (run loop only) | done (`5983b6c`) |
| — | `--render` flag for visual sanity | done (`1507af1`) |
| 2 + 5 | Hardware ownership to Controller, dual-mode collapse | done (`aa92a3f`) |
| 3 + 4 | Formalize FSM, add `DAMP` state | done (`f80806a`) |
| 7 | Update FAR-pi extensions, rewrite skipped input tests | not started |
| 8 | `PolicyProtocol` — make state→action a first-class type | not started |

Steps 1–4 already landed and are described at the bottom of this doc as
"what's true today." The body of the doc describes Step 8, which is
where the architecture is actually heading.

## Problem (the one Step 8 closes)

`BasePolicy` is doing two unrelated things:

1. **State→action mapping** — the actual control law. The hot path of
   `policy_action()`. This is what every kind of "active behaviour"
   (locomotion, WBT, damping, init ramp, stiff hold) shares.
2. **Observation pipeline** — obs scaling, history buffers, ONNX
   wiring. An implementation detail of the *learned* policies. Damping
   and init don't need any of it.

Today every variant of (1) inherits the apparatus of (2) by being a
subclass of `BasePolicy`. That's why DAMP had to land as a Controller
flag (`_damp_active`) rather than as a peer of `LocomotionPolicy` —
making it a `BasePolicy` subclass would have forced it to drag along
ONNX init, obs config, and the rest.

Step 8 fixes this by promoting (1) to a `PolicyProtocol` and letting
each impl decide how it wants to satisfy it.

## The protocol

```python
class PolicyProtocol(Protocol):
    """Maps robot state to a low-level command."""
    name: str

    def act(self, ctx: Controller, state: np.ndarray) -> Command:
        """One tick: state → low-level command. The hot path."""

    def on_activate(self, ctx: Controller) -> None:
        """Called when this policy becomes the active one.
        KP/KD push, q_hold capture, gait-phase reset, init counter reset."""

    def on_deactivate(self, ctx: Controller) -> None:
        """Called when this policy stops being active. Most impls are no-op."""

    def apply_velocity(self, vc: VelCmd) -> None:
        """Locomotion gates by stand_command; WBT ignores; damping ignores."""

    def apply_command(self, cmd: StateCommand) -> bool:
        """Handle a policy-specific command. Returns True if handled,
        False to fall through to the Controller's builtin dispatch."""
```

Five members. Symmetric verb shape (`apply_velocity` / `apply_command`).
`apply_command` returns `bool` so the Controller can fall back to
builtin handlers for commands the active policy doesn't bind.

## `Command` dataclass

```python
@dataclass
class Command:
    q: np.ndarray                          # (N,) target joint positions
    dq: np.ndarray | None = None
    tau: np.ndarray | None = None
    kp_override: np.ndarray | None = None
    kd_override: np.ndarray | None = None
    dof_pos_latest: np.ndarray | None = None
```

Same six fields as `interface.send_low_command`, reified. Controller is
the only caller of `send_low_command`. Every `act()` returns a `Command`.

## Concrete implementations

```python
class OnnxLocomotionPolicy(OnnxBasePolicy):
    """Today's LocomotionPolicy. Inherits OnnxBasePolicy for obs/ONNX."""
    name = "locomotion"
    def act(self, ctx, state):
        if self.use_phase: self.update_phase_time()
        action = self._rl_inference(state)
        return Command(q=action + self.default_dof_angles + self.joint_offsets)
    def on_activate(self, ctx):
        self._resolve_control_gains()
        self.phase = np.array([[0.0, np.pi]])
    def apply_command(self, cmd):
        if   cmd is StateCommand.STAND_TOGGLE:  self._toggle_stand(); return True
        elif cmd is StateCommand.ZERO_VELOCITY: self._zero_velocity(); return True
        elif cmd in (StateCommand.WALK, StateCommand.STAND):
            self._set_stand(cmd is StateCommand.STAND); return True
        return False
    def apply_velocity(self, vc):
        ...  # same gating-by-stand_command as today

class OnnxWBTPolicy(OnnxBasePolicy):
    """Today's WholeBodyTrackingPolicy."""
    name = "wbt"
    def apply_command(self, cmd):
        if cmd is StateCommand.START_MOTION_CLIP:
            self._handle_start_motion_clip(); return True
        return False

class DampingPolicy:
    """Hold last observed q with policy KP/KD. ~40 LOC, no inheritance."""
    name = "damping"
    def __init__(self, kp_scale=1.0, kd_scale=1.0):
        self.kp_scale, self.kd_scale, self._q_hold = kp_scale, kd_scale, None
    def on_activate(self, ctx):    self._q_hold = None
    def on_deactivate(self, ctx):  self._q_hold = None
    def apply_velocity(self, vc):  pass
    def apply_command(self, cmd):  return False
    def act(self, ctx, state):
        n = ctx.num_dofs
        if self._q_hold is None: self._q_hold = state[7:7+n].copy()
        return Command(
            q=self._q_hold + ctx.joint_offsets,
            dq=np.zeros(n), tau=np.zeros(n),
            dof_pos_latest=state[7:7+n],
            kp_override=ctx.motor_kp * self.kp_scale,
            kd_override=ctx.motor_kd * self.kd_scale,
        )

class InitPolicy:
    """Interpolate from current dof_pos → target_q over n_steps. ~30 LOC."""
    name = "init"
    def __init__(self, target_q, n_steps=500):
        self.target_q, self.n_steps = np.asarray(target_q), n_steps
        self._counter, self._q0 = 0, None
    def on_activate(self, ctx):
        self._counter = 0
        self._q0 = ctx.interface.get_low_state()[0, 7:7+ctx.num_dofs].copy()
    def is_done(self): return self._counter >= self.n_steps
    def apply_command(self, cmd): return False
    def apply_velocity(self, vc): pass
    def act(self, ctx, state):
        alpha = min(self._counter / self.n_steps, 1.0); self._counter += 1
        q = self._q0 + (self.target_q - self._q0) * alpha
        return Command(q=q + ctx.joint_offsets, dof_pos_latest=state[7:7+ctx.num_dofs])

class StiffHoldPolicy:
    """WBT's startup stiff hold. Replaces the _stiff_hold_active flag. ~20 LOC."""
    name = "stiff_hold"
    def __init__(self, q, kp, kd):
        self.q, self.kp, self.kd = np.asarray(q), np.asarray(kp), np.asarray(kd)
    def on_activate(self, ctx):  pass
    def on_deactivate(self, ctx): pass
    def apply_command(self, cmd): return False
    def apply_velocity(self, vc): pass
    def act(self, ctx, state):
        n = ctx.num_dofs
        return Command(q=self.q + ctx.joint_offsets,
                       kp_override=self.kp, kd_override=self.kd,
                       dof_pos_latest=state[7:7+n])
```

Lightweight policies (`DampingPolicy`, `InitPolicy`, `StiffHoldPolicy`)
implement `PolicyProtocol` directly — they don't pay for the obs/ONNX
machinery they don't use. ONNX-based policies inherit from
`OnnxBasePolicy` (renamed from `BasePolicy`) for that machinery.

## Controller

```python
class Controller:
    def __init__(self, policies, initial, *, interface, velocity_input,
                 command_provider, rate, logger=None):
        self.policies = policies
        self.active = policies[initial]
        self.interface, self.velocity_input = interface, velocity_input
        self.command_provider, self.rate = command_provider, rate
        self.logger = logger or _default_logger
        self.active.on_activate(self)

    # Convenience accessors for policies (they take ctx, not raw config):
    @property
    def num_dofs(self):  return self._num_dofs
    @property
    def motor_kp(self):  return np.asarray(self._robot_config.motor_kp, dtype=np.float64)
    @property
    def motor_kd(self):  return np.asarray(self._robot_config.motor_kd, dtype=np.float64)
    @property
    def joint_offsets(self): return self._joint_offsets

    def transition_to(self, name: str) -> None:
        if name == self.active.name: return
        self.active.on_deactivate(self)
        self.active = self.policies[name]
        self.active.on_activate(self)
        self.logger.info(f"Active policy: {name}")

    def step(self) -> None:
        vc = self.velocity_input.poll_velocity()
        if vc is not None: self.active.apply_velocity(vc)

        for cmd in self.command_provider.poll_commands():
            if not self.active.apply_command(cmd):
                self._builtin_dispatch(cmd)

        state = self.interface.get_low_state()[0]
        command = self.active.act(self, state)
        self._send(command)

    def _builtin_dispatch(self, cmd):
        if   cmd is StateCommand.START:  self.transition_to(self._default_run_policy)
        elif cmd is StateCommand.STOP:   self.transition_to("damping")
        elif cmd is StateCommand.INIT:   self.transition_to("init")
        elif cmd is StateCommand.DAMP:   self.transition_to("damping")
        elif cmd is StateCommand.KILL:   sys.exit(0)
        elif cmd in STATE_COMMAND_TO_POLICY_INDEX: ...   # multi-model select
        elif cmd is StateCommand.SWITCH_MODE: self._cycle_run_policies()
        elif cmd is StateCommand.NEXT_POLICY: ...
        elif cmd in {KP_UP, KP_DOWN, ...}: self._adjust_kp(cmd)

    def _send(self, c: Command):
        n = self._num_dofs
        zeros = np.zeros(n)
        self.interface.send_low_command(
            c.q, c.dq if c.dq is not None else zeros,
            c.tau if c.tau is not None else zeros,
            c.dof_pos_latest,
            kp_override=c.kp_override, kd_override=c.kd_override,
        )

    def run(self):
        try:
            while True:
                self.step()
                self.rate.sleep()
        except KeyboardInterrupt: pass
```

Controller is ~80 LOC. The 5-way `if get_ready/use_policy/_stiff_hold/...`
branch is gone — each branch is now its own policy.

## Adam's use case in two lines

```python
# holosoma_service (daemon):
controller = Controller(
    policies={
        "damping":    DampingPolicy(),
        "locomotion": OnnxLocomotionPolicy(config=cfg, interface=interface),
        "init":       InitPolicy(target_q=cfg.robot.default_dof_angles),
    },
    initial="damping",        # robot starts energized at default pose
    interface=interface, velocity_input=vel_in, command_provider=cmd_in, rate=rate,
)
controller.run()

# teleop_app (client) sends `transition_to("locomotion")` over IPC.
# Releases handle? Service catches the disconnect, calls transition_to("damping").
# Robot never goes slack.
```

Dual-mode = `policies={"primary": ..., "secondary": ...}` and `SWITCH_MODE`
cycles between policy keys.

## What gets deleted by Step 8

- `DualModePolicy` class (~140 LOC).
- `BasePolicy._handle_start_policy / _handle_stop_policy / _handle_init_state / _handle_damp_state`.
- `use_policy_action`, `get_ready_state`, `_stiff_hold_active`, `init_count` flags.
- `Controller.state` property, `set_state` writethrough, `_damp_active`,
  `_damp_q`, `_publish_damp_command`, `ControllerState` enum.
- `policy_action()`'s 5-way branching.
- The `_dispatch_command` lambda-patching in `DualModePolicy.bind_controller`.

## Layout

```
holosoma_inference/
├── controllers/                    NEW — orchestrator + protocol
│   ├── __init__.py
│   ├── controller.py               Controller class (moved from top-level)
│   └── protocol.py                 PolicyProtocol, Command
├── policies/                       all PolicyProtocol implementations
│   ├── base.py                     OnnxBasePolicy (renamed from BasePolicy)
│   ├── locomotion.py               OnnxLocomotionPolicy
│   ├── wbt.py                      OnnxWBTPolicy
│   ├── damping.py                  NEW — DampingPolicy (~40 LOC)
│   ├── init_ramp.py                NEW — InitPolicy (~30 LOC)
│   └── stiff_hold.py               NEW — StiffHoldPolicy (~20 LOC)
```

`BasePolicy = OnnxBasePolicy` deprecation alias for one release cycle.
The current `holosoma_inference/controller.py` moves to
`holosoma_inference/controllers/controller.py`; the top-level file
becomes a deprecation shim re-exporting `Controller` for one cycle.

## Migration order for Step 8

1. Add `PolicyProtocol` and `Command` (no consumers yet).
2. Rename `BasePolicy → OnnxBasePolicy`. Make it conform to the protocol
   by adding `act` (wraps `policy_action()`), `on_activate` (wraps
   `_handle_start_policy`), `apply_command`, `apply_velocity`. Keep the
   old methods as private helpers for one step. Harness still passes.
3. Add `DampingPolicy`, `InitPolicy`, `StiffHoldPolicy` as new files.
4. Rewrite `Controller` to drive policies by protocol. Delete
   `set_state`, `_damp_active`, `_publish_damp_command`, `ControllerState`.
5. Delete `DualModePolicy`. Rewrite `run_policy.py` to build the
   policies dict from the config (locomotion + damping always; init if
   the user has an init-pose; secondary if dual-mode).
6. Delete the legacy flags from `OnnxBasePolicy`.
7. Rewrite the harness to construct policies dict directly.
8. Rewrite `inputs/tests/{test_factory,test_providers,test_dual_mode}.py`
   against the new protocol.

The transition is non-trivial but each step is independently testable
against the sim2sim harness.

## Out of scope for Step 8

- Mode chaining ("after INIT completes, auto-transition to locomotion").
- Declared transition graph with `(from, to)` validation.
- Per-policy persistent state across activations (today each policy
  resets on `on_activate`).

---

# What's true today (after Steps 1–4)

These are the abstractions Step 8 builds on / replaces.

## States (today)

```python
class ControllerState(Enum):
    IDLE         # no commands sent
    INIT         # interpolating to default pose
    DAMP         # holds last q with low gains
    STIFF_HOLD   # WBT's startup hold
    RUN_POLICY   # policy_action() drives the robot
```

`Controller.state` is a derived property — reads back the legacy flags
(`use_policy_action`, `get_ready_state`, `_stiff_hold_active`) and the
controller-side `_damp_active`. `Controller.set_state(s)` writes through
to those flags atomically. Step 8 deletes both.

## Controller API (today)

```python
class Controller:
    def __init__(self, policy, interface, velocity_input, command_provider,
                 rate, logger=None, use_joystick=False, use_keyboard=False): ...
    state: ControllerState                   # derived property
    def set_state(s: ControllerState): ...
    def set_policy(p: BasePolicy): ...       # used by DualModePolicy
    def step(): ...                          # one rl_rate tick
    def run(): ...                           # while True: step(); rate.sleep()
```

## `DAMP` state wires up via `StateCommand.DAMP`

Keyboard `\\`, joystick `B+X`. Controller captures `q` on entry from
`interface.get_low_state()`, then publishes `send_low_command(q_hold,
kp_override=kp, kd_override=kd)` every tick until exited.

## Dual-mode (today, Steps 2+5)

```python
class DualModePolicy:
    def __init__(self, primary_config, secondary_config, interface):
        self.primary = ...; self.secondary = ...
    def bind_controller(self, controller):
        # Inject SWITCH_MODE → self._handle_mode_switch into command provider
        # Patch each policy's _dispatch_command with a SWITCH_MODE intercept
    def run(self): self.controller.run()
```

The patched-`_dispatch_command` lambdas are the worst remaining smell —
Step 8's protocol-based approach removes them entirely.
