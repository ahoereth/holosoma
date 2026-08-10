"""Unit tests for WBT self-collision-aware spawn resampling (MotionCommand).

Two layers:
- ``no_sim``: exercise the real ``_resample_self_colliding_spawns`` rejection-loop logic with
  stubbed frame-sampling / pose-assembly / collision-check, so index alignment, the shrink of the
  colliding subset, the redraw-until-clean behavior, and the hard error are covered without a sim.
- ``mujoco``: build the real g1 MuJoCo collision model via the same helper the feature uses and
  confirm a rest pose is collision-free while an arms-inward pose self-collides.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from holosoma.managers.command.terms.wbt import MotionCommand  # noqa: E402


# --------------------------------------------------------------------------------------------------
# no_sim: rejection-loop logic with stubbed helpers
# --------------------------------------------------------------------------------------------------
def _make_command(max_attempts: int, colliding_frames: set[int], frame_plan: dict[int, list[int]]):
    """Build a MotionCommand shell (no __init__) wired for _resample_self_colliding_spawns.

    Model: each env's "dof_pos" is a single scalar = its assigned motion frame. A frame collides
    iff it is in ``colliding_frames``. ``frame_plan[env_id]`` is the sequence of frames that env
    will be (re)assigned on successive ``_sample_frames`` calls (index 0 is the initial frame; the
    initial targets are built by the caller, so index 1 onward are the redraws).
    """
    cmd = object.__new__(MotionCommand)
    cmd.device = torch.device("cpu")
    cmd.motion_cfg = types.SimpleNamespace(max_self_collision_resample_attempts=max_attempts)
    # motion_ids just needs to be indexable by env id for the error message.
    cmd.motion_ids = torch.zeros(64, dtype=torch.long)
    cmd.time_steps = torch.zeros(64, dtype=torch.long)
    # Observability state that _resample_self_colliding_spawns updates (normally set in
    # _setup_self_collision_check).
    cmd.metrics = {}
    cmd._sc_total_resampled = 0
    cmd._sc_resets_with_collision = 0

    # per-env cursor into frame_plan (starts at 1: index 0 was the initial assignment)
    cursor = {e: 1 for e in frame_plan}

    def _sample_frames(env_ids):
        for e in env_ids.tolist():
            seq = frame_plan[e]
            idx = min(cursor[e], len(seq) - 1)
            cmd.time_steps[e] = seq[idx]
            cursor[e] += 1

    def _assemble_spawn_targets(env_ids):
        frames = cmd.time_steps[env_ids].clone().float().unsqueeze(1)  # (k, 1)
        return {"dof_pos": frames, "tag": frames * 100.0}

    def _check_self_collision(dof_pos):
        frames = dof_pos.squeeze(1).long().tolist()
        return torch.tensor([f in colliding_frames for f in frames], dtype=torch.bool)

    cmd._sample_frames = _sample_frames
    cmd._assemble_spawn_targets = _assemble_spawn_targets
    cmd._check_self_collision = _check_self_collision
    return cmd


def _initial_targets(cmd, env_ids, initial_frames):
    """Set the initial frame per env and build the initial targets dict (parallel to env_ids)."""
    for e, f in zip(env_ids.tolist(), initial_frames):
        cmd.time_steps[e] = f
    frames = torch.tensor(initial_frames, dtype=torch.float32).unsqueeze(1)
    return {"dof_pos": frames, "tag": frames * 100.0}


@pytest.mark.no_sim
def test_no_collision_is_noop():
    cmd = _make_command(max_attempts=5, colliding_frames=set(), frame_plan={0: [10], 1: [20]})
    env_ids = torch.tensor([0, 1])
    targets = _initial_targets(cmd, env_ids, [10, 20])
    cmd._resample_self_colliding_spawns(env_ids, targets)
    # unchanged
    assert targets["dof_pos"].squeeze(1).tolist() == [10.0, 20.0]
    assert targets["tag"].squeeze(1).tolist() == [1000.0, 2000.0]


@pytest.mark.no_sim
def test_single_env_redraws_until_clean():
    # env 0 starts on colliding frame 3, redraws hit 4 (colliding) then 7 (clean).
    cmd = _make_command(max_attempts=5, colliding_frames={3, 4}, frame_plan={0: [3, 4, 7]})
    env_ids = torch.tensor([0])
    targets = _initial_targets(cmd, env_ids, [3])
    cmd._resample_self_colliding_spawns(env_ids, targets)
    assert targets["dof_pos"].squeeze(1).tolist() == [7.0]
    # tag key must scatter in lockstep with dof_pos
    assert targets["tag"].squeeze(1).tolist() == [700.0]


@pytest.mark.no_sim
def test_only_colliding_subset_is_redrawn_and_alignment_holds():
    # env 0 clean (5); env 1 colliding (3)->clean(9); env 2 clean (6). Only env 1 should change,
    # and it must land at its own plan value, not another env's.
    cmd = _make_command(
        max_attempts=5,
        colliding_frames={3},
        frame_plan={0: [5], 1: [3, 9], 2: [6]},
    )
    env_ids = torch.tensor([0, 1, 2])
    targets = _initial_targets(cmd, env_ids, [5, 3, 6])
    cmd._resample_self_colliding_spawns(env_ids, targets)
    assert targets["dof_pos"].squeeze(1).tolist() == [5.0, 9.0, 6.0]
    assert targets["tag"].squeeze(1).tolist() == [500.0, 900.0, 600.0]


@pytest.mark.no_sim
def test_raises_when_motion_always_collides():
    # every frame env 0 ever draws collides -> cap exhausted -> RuntimeError naming the env.
    cmd = _make_command(max_attempts=3, colliding_frames={1, 2, 3, 4}, frame_plan={0: [1, 2, 3, 4]})
    env_ids = torch.tensor([0])
    targets = _initial_targets(cmd, env_ids, [1])
    with pytest.raises(RuntimeError, match="Self-collision resampling failed"):
        cmd._resample_self_colliding_spawns(env_ids, targets)


@pytest.mark.no_sim
def test_partial_failure_reports_only_stuck_env():
    # env 0 escapes (3->8), env 1 never escapes. Error must name env 1, not env 0.
    cmd = _make_command(
        max_attempts=3,
        colliding_frames={3, 1, 2, 5, 9},
        frame_plan={0: [3, 8], 1: [1, 2, 5, 9]},
    )
    env_ids = torch.tensor([0, 1])
    targets = _initial_targets(cmd, env_ids, [3, 1])
    with pytest.raises(RuntimeError, match=r"\[1\]"):
        cmd._resample_self_colliding_spawns(env_ids, targets)


# --------------------------------------------------------------------------------------------------
# mujoco: real g1 collision model built via the feature's own helper
# --------------------------------------------------------------------------------------------------
@pytest.mark.mujoco
def test_g1_urdf_self_collision_detection():
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    from holosoma.config_values.robot import g1_29dof
    from holosoma.managers.command.terms.wbt import _load_urdf_with_mesh_assets
    from holosoma.utils.path import resolve_asset_path

    asset = g1_29dof.asset
    urdf = resolve_asset_path(asset.urdf_file, asset.asset_root)
    text, assets = _load_urdf_with_mesh_assets(urdf)
    model = mujoco.MjModel.from_xml_string(text, assets=assets)
    assert model.ngeom > 0
    data = mujoco.MjData(model)

    # rest pose: no self-collision
    mujoco.mj_forward(model, data)
    assert data.ncon == 0

    # arms cranked inward: self-collision
    for name in ("left_shoulder_roll_joint", "right_shoulder_roll_joint",
                 "left_elbow_joint", "right_elbow_joint"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"joint {name} missing from g1 URDF"
        data.qpos[model.jnt_qposadr[jid]] = 1.6
    mujoco.mj_forward(model, data)
    assert data.ncon > 0


@pytest.mark.mujoco
def test_check_self_collision_ignores_world_contacts():
    """_check_self_collision must count only robot-vs-robot contacts, not robot-vs-ground.

    Regression guard: the initial implementation used raw ``data.ncon > 0``, which flagged the
    ground contact from a ``*_contact.urdf`` ground plane (the robot's free base sits at the
    identity and clips the floor) as a spurious "self-collision". We wrap the g1 URDF in a scene
    that adds a ground plane, then confirm: a rest pose (feet clipping the ground) reads NOT
    self-colliding, while an arms-inward pose reads self-colliding.
    """
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    from holosoma.config_values.robot import g1_29dof
    from holosoma.managers.command.terms.wbt import _load_urdf_with_mesh_assets
    from holosoma.utils.path import resolve_asset_path

    asset = g1_29dof.asset
    urdf = resolve_asset_path(asset.urdf_file, asset.asset_root)
    text, assets = _load_urdf_with_mesh_assets(urdf)
    # Add a world ground plane (mirrors what a *_contact.urdf carries), raised to z=0.2 so the
    # identity-base robot clips it and always produces a robot-vs-world contact to be filtered out.
    spec = mujoco.MjSpec.from_string(text, assets=assets)
    spec.worldbody.add_geom(
        name="_test_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], pos=[0, 0, 0.2]
    )
    model = spec.compile()

    cmd = object.__new__(MotionCommand)
    cmd._sc_model = model
    cmd._sc_data = mujoco.MjData(model)
    dof_names = list(g1_29dof.dof_names)
    addr, cols = [], []
    for c, name in enumerate(dof_names):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            addr.append(int(model.jnt_qposadr[j]))
            cols.append(c)
    cmd._sc_qpos_addr = np.array(addr, dtype=np.int64)
    cmd._sc_motion_cols = np.array(cols, dtype=np.int64)
    cmd._sc_geom_is_robot = np.array(
        [model.geom_bodyid[g] != 0 for g in range(model.ngeom)], dtype=bool
    )

    # Sanity: the injected ground plane contacts the robot at the rest pose.
    mujoco.mj_forward(model, cmd._sc_data)
    assert cmd._sc_data.ncon > 0, "expected a robot-vs-ground contact from the injected plane"

    rest = torch.zeros(1, len(dof_names))
    arms_in = torch.zeros(1, len(dof_names))
    for i, name in enumerate(dof_names):
        low = name.lower()
        if "shoulder_roll" in low or "elbow" in low:
            arms_in[0, i] = 1.6

    # Rest pose: only ground contact -> filtered out -> NOT self-colliding.
    # Arms-inward: genuine link-vs-link overlap -> self-colliding (ground still ignored).
    flags = cmd._check_self_collision(torch.cat([rest, arms_in], dim=0))
    assert flags.tolist() == [False, True]
