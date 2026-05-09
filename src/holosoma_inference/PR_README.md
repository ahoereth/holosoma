# Controller refactor — Step 0 + Step 1

Branch: `dev/tomasz/controller`

## What's in this PR

| Step | Commit | Status |
|---|---|---|
| 0 — Sim2sim test harness + design doc | `846c299` | done |
| 1 — Extract `Controller` (run loop only, no behavior change) | next | done |
| 2 — Move hardware ownership to Controller | future PR | not started |
| 3 — Formalize FSM (replace flags with `ControllerState`) | future PR | not started |
| 4 — Add `DAMP` state | future PR | not started |
| 5 — Collapse `DualModePolicy` to a swap | future PR | not started |

Steps 2–5 will follow as separate PRs because each one touches all policy
subclasses and warrants its own review.

## Design

`docs/controller-design.md` — the one-pager. TL;DR: a `Controller` orchestrates
a `BasePolicy` through a 5-state FSM (`IDLE/INIT/DAMP/STIFF_HOLD/RUN_POLICY`).
Step 1 only carves out the run loop; ownership migration happens in later steps.

## How to verify (sim2sim)

### Headless harness (~2 s)

```bash
source scripts/source_inference_setup.sh   # or any env with mujoco + holosoma_inference
cd src/holosoma_inference
PYTHONPATH=. python -m tests.sim2sim.harness
```

Expected:

```
[OK] pelvis final=0.768 m, min=0.756 m, steps=500
```

Run before the change (checkout `846c299`) and after — the result must
match within rounding.

### Pytest

```bash
cd src/holosoma_inference
PYTHONPATH=. python -m pytest tests/sim2sim/ -v
```

### Live sim2sim (interactive)

Same as `docs/workflows/sim-to-sim-locomotion.md` — no commands changed.
Two terminals:

```bash
# terminal 1 — simulator
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof

# terminal 2 — policy
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.no-use-joystick --task.interface lo
```

In MuJoCo: `8` to lower gantry, `9` to remove. In policy terminal: `]` to start,
`=` to walk, then `w/a/s/d` for velocity. Robot should walk identically to
before this PR.

## What changed

- `holosoma_inference/controller.py` — new file. `Controller` + `ControllerState`.
- `holosoma_inference/policies/base.py` — `BasePolicy.run()` now delegates to
  `Controller(self).run()`. Three lines instead of thirty. The body of the
  former loop is in `Controller.step()` and `Controller.run()`.
- `tests/sim2sim/` — new directory with the headless harness, MuJoCo
  interface stub, and a pytest entry.

## What did NOT change

- Hardware ownership: `interface`, `_velocity_input`, `_command_provider`,
  `rate`, `latency_tracker` still live on `BasePolicy`. Step 2 moves them.
- `DualModePolicy.run()` — still has its own loop. Step 5 collapses it.
- Any policy subclass (`LocomotionPolicy`, `WholeBodyTrackingPolicy`).
- `run_policy.py` — unchanged. The `policy.run()` call now goes through
  Controller transparently.

## Known limitations

- The harness is a coarse gate: pelvis-height threshold only. It will not
  catch subtle observation-history bugs, KP/KD swap mismatches, or
  dual-mode SWITCH_MODE regressions. Live sim2sim and on-robot testing
  are still required before each future step lands.
- The harness uses the `_shared_hardware_source` injection pattern that
  Step 2 will delete. The harness will need to be rewritten in Step 2's
  PR to construct a Controller directly.

## FAR-pi extensions

`~/projects/FAR-pi/holosoma_extensions/` overrides `BasePolicy.rl_inference`,
calls private base methods as unbound functions, and uses
`_shared_hardware_source`. Step 1 leaves all of those untouched. Steps
2–3 will break them — Step 7 (separate PR) will update FAR-pi.

## Baseline result

| Metric | Value |
|---|---|
| Pelvis final z | 0.768 m |
| Pelvis min z | 0.756 m |
| Steps | 500 (10 s @ 50 Hz) |
| Verdict | OK |

Identical between Step 0 (`846c299`) and Step 1.
