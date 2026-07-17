"""Unit tests for the service node's target-source attach + policy iteration.

These cover the WBT-unification contract without spinning ROS or loading ONNX:

* ``_attach_target_source`` sets ``_target_source`` on policies that expose it
  (WBT family) and leaves locomotion-style policies (no such attribute) untouched
  — so the always-on ``CmdDense`` subscription is inert for loco.
* ``_iter_policies`` yields the single policy normally and every policy from a
  MultiModePolicy-shaped object.

``service_node`` imports numpy/rclpy/holosoma_msgs at module load, so skip
cleanly when those aren't present (host env); the bazel pytest_test target has
them.
"""

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("rclpy")
pytest.importorskip("holosoma_msgs")

from holosoma_inference.policies.dual_mode import MultiModePolicy
from holosoma_service.policy_control import service_node


class _WBTLikePolicy:
    """Exposes ``_target_source`` (WBT family) — should be attached."""

    def __init__(self):
        self._target_source = None


class _LocoLikePolicy:
    """No ``_target_source`` attribute — should be left untouched."""


class _IO:
    """Stand-in for ServiceIONode (only identity matters for attach)."""


def test_attach_sets_target_source_on_wbt_like():
    policy = _WBTLikePolicy()
    io = _IO()
    service_node._attach_target_source(policy, io)
    assert policy._target_source is io


def test_attach_skips_loco_like_without_target_source():
    policy = _LocoLikePolicy()
    io = _IO()
    service_node._attach_target_source(policy, io)
    assert not hasattr(policy, "_target_source")


def test_iter_policies_single():
    policy = _LocoLikePolicy()
    assert list(service_node._iter_policies(policy)) == [policy]


def test_iter_policies_multi_mode_yields_all_policies():
    mm = object.__new__(MultiModePolicy)
    mm.policies = [_WBTLikePolicy(), _LocoLikePolicy()]
    assert list(service_node._iter_policies(mm)) == mm.policies


def test_attach_target_source_on_multi_mode_attaches_only_wbt_member():
    mm = object.__new__(MultiModePolicy)
    mm.policies = [_WBTLikePolicy(), _LocoLikePolicy()]
    io = _IO()
    service_node._attach_target_source(mm, io)
    assert mm.policies[0]._target_source is io
    assert not hasattr(mm.policies[1], "_target_source")


def test_resolve_extra_policy_uses_registry(monkeypatch):
    from holosoma_inference.config.config_values.inference import g1_29dof_loco

    monkeypatch.setattr(service_node, "load_plugins", lambda: None)
    monkeypatch.setattr(service_node, "INFERENCE_REGISTRY", {"test-loco": g1_29dof_loco})

    [resolved] = service_node._resolve_extra_policies(
        ["inference:test-loco"],
        [json.dumps({"model_path": "/models/extra.onnx"})],
    )

    assert resolved.task.model_path == "/models/extra.onnx"
    assert resolved.secondary is None
    assert resolved.policies == ()


def test_resolve_extra_policy_rejects_unknown_registry_key(monkeypatch):
    monkeypatch.setattr(service_node, "load_plugins", lambda: None)
    monkeypatch.setattr(service_node, "INFERENCE_REGISTRY", {})

    with pytest.raises(SystemExit, match="unknown inference key"):
        service_node._resolve_extra_policies(["inference:missing"], [])
