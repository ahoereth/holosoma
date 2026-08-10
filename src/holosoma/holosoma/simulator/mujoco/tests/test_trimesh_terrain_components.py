"""MuJoCo collides a mesh geom as its CONVEX HULL, so concave terrain needs one geom per component.

A stepped/stair course is concave. Emitted as a single mesh geom, its collision surface is the hull
over every vertex — a smooth ramp spanning the whole course that sits *above* the true ground near
the start, so a robot spawned there begins embedded in the terrain. Nothing errors: the mesh loads,
the geom is created, and the robot simply sinks.

These tests work on the shipped terrain mesh directly (no simulator needed) and assert the property
``MujocoSceneManager._create_trimesh`` relies on: per-component hulls reproduce the true surface,
while a single hull does not.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from holosoma.utils.path import resolve_data_file_path

pytestmark = pytest.mark.no_sim

TERRAIN_OBJ = "holosoma/data/terrains/terrain.obj"
# The spawn point run_sim uses (env origin is the world origin; robot z comes from init_state).
SPAWN_XY = (0.0, 0.0)


def _load_terrain() -> trimesh.Trimesh:
    mesh = trimesh.load(resolve_data_file_path(TERRAIN_OBJ), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def _surface_z(meshes: list[trimesh.Trimesh], x: float, y: float) -> float | None:
    """Highest downward ray hit at (x, y) across ``meshes`` — the collision surface height."""
    best: float | None = None
    for mesh in meshes:
        hits, _, _ = mesh.ray.intersects_location(
            ray_origins=np.array([[x, y, 50.0]]), ray_directions=np.array([[0.0, 0.0, -1.0]])
        )
        if len(hits):
            top = float(hits[:, 2].max())
            best = top if best is None else max(best, top)
    return best


def test_terrain_mesh_is_concave_and_splits_into_components() -> None:
    """The premise: the shipped course is multi-body, so splitting it is meaningful."""
    mesh = _load_terrain()
    components = [c for c in mesh.split() if len(c.vertices) and len(c.faces)]
    assert len(components) > 1, "a single-component mesh would make the per-component split a no-op"


def test_single_convex_hull_would_bury_the_spawn_point() -> None:
    """Documents the failure this split exists to prevent — the regression is silent otherwise."""
    mesh = _load_terrain()
    true_z = _surface_z([mesh], *SPAWN_XY)
    hull_z = _surface_z([mesh.convex_hull], *SPAWN_XY)

    assert true_z is not None and hull_z is not None
    # The hull lifts the ground at the start of the course well above the real surface.
    assert hull_z > true_z + 0.1, (
        f"expected the whole-mesh hull to sit above the true ground at {SPAWN_XY}; got hull={hull_z} vs true={true_z}"
    )


@pytest.mark.parametrize("x", [0.0, 2.0, 5.0, 10.0, 20.0, 70.0])
def test_per_component_hulls_reproduce_the_true_surface(x: float) -> None:
    """Each component is convex, so per-component hulls collide as the real geometry does."""
    mesh = _load_terrain()
    hulls = [c.convex_hull for c in mesh.split() if len(c.vertices) and len(c.faces)]

    true_z = _surface_z([mesh], x, SPAWN_XY[1])
    hull_z = _surface_z(hulls, x, SPAWN_XY[1])

    assert true_z is not None and hull_z is not None
    assert hull_z == pytest.approx(true_z, abs=1e-6), f"collision surface differs from the mesh at x={x}"


def test_spawn_height_clears_the_collision_surface() -> None:
    """The robot's ``init_state`` z must sit above the terrain under the spawn point."""
    from holosoma.config_values.robot import ROBOT_REGISTRY

    mesh = _load_terrain()
    hulls = [c.convex_hull for c in mesh.split() if len(c.vertices) and len(c.faces)]
    surface_z = _surface_z(hulls, *SPAWN_XY)
    spawn_z = ROBOT_REGISTRY["g1_29dof"].init_state.pos[2]

    assert surface_z is not None
    # Pelvis spawn height, less standing leg length, still has to clear the ground.
    assert spawn_z > surface_z, f"g1_29dof spawns at z={spawn_z} but the terrain there is z={surface_z}"
