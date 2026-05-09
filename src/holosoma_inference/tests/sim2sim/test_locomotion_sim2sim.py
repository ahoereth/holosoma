"""Pytest entry point for the locomotion sim2sim harness."""

from __future__ import annotations

import pytest

from tests.sim2sim.harness import run_harness


@pytest.mark.slow
def test_g1_loco_does_not_fall():
    """G1 locomotion policy keeps pelvis above 0.3 m for 10 s of sim time."""
    result = run_harness(duration_s=10.0)
    assert not result.fell, result.summary()
    # Final pose should still be roughly upright
    assert result.final_pelvis_z > 0.6, result.summary()
