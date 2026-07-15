"""Eval callback that records per-step trajectory data to an NPZ file.

Records joint positions, velocities, torques, body poses, and root state
for later visualization with viser_eval_viewer.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import RecordingConfig
from holosoma.utils.safe_torch_import import torch


class EvalRecordingCallback(RLEvalCallback):
    """Records per-step data during evaluation and saves to .npz on completion."""

    def __init__(
        self,
        config: RecordingConfig,
        training_loop: Any = None,
    ):
        super().__init__(config, training_loop)

        def _resolve(path: str) -> str:
            if not path.endswith(".npz"):
                path += ".npz"
            if training_loop is not None and hasattr(training_loop, "log_dir"):
                path = str(Path(training_loop.log_dir) / path)
            return path

        # Multi-task mode: one env per task, one NPZ per task (single-env format).
        # Single-env mode (default): one env_id -> one NPZ. Both are modeled as a
        # list of (env_id, output_path, task_name, command) "targets" so the
        # per-step extraction code is shared.
        self._tasks = self._load_tasks(getattr(config, "tasks_json", ""))
        if self._tasks:
            output_dir = Path(_resolve(config.output_path)).parent
            self.targets = [
                {
                    "env_id": i,
                    "output_path": str(output_dir / f"{task['name']}.npz"),
                    "command": [task.get("vx", 0.0), task.get("vy", 0.0), task.get("yaw", 0.0)],
                }
                for i, task in enumerate(self._tasks)
            ]
        else:
            self.targets = [{"env_id": config.env_id, "output_path": _resolve(config.output_path), "command": None}]

        # Per-target buffers/metadata, keyed by env_id.
        self._buffers: dict[int, dict[str, list[np.ndarray]]] = {t["env_id"]: {} for t in self.targets}
        self._metadata: dict[int, dict[str, Any]] = {t["env_id"]: {} for t in self.targets}
        self._step_count = 0

    @staticmethod
    def _load_tasks(tasks_json: str) -> list[dict]:
        if not tasks_json:
            return []
        return json.loads(Path(tasks_json).read_text())

    def _get_env(self):
        """Get the unwrapped BaseTask environment."""
        return self.training_loop._unwrap_env()

    def _save(self) -> None:
        """Save each target's recorded data to its own single-env-format NPZ."""
        if self._step_count == 0:
            return

        for target in self.targets:
            env_id = target["env_id"]
            arrays: dict[str, np.ndarray] = {}
            for name, values in self._buffers[env_id].items():
                if values:
                    arrays[name] = np.stack(values, axis=0)
            arrays["_metadata_json"] = np.array(json.dumps(self._metadata[env_id]))

            path = Path(target["output_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(path), **arrays)

            channel_summary = ", ".join(
                f"{name}{list(arr.shape)}" for name, arr in arrays.items() if name != "_metadata_json"
            )
            logger.info(
                f"EvalRecordingCallback: saved {self._step_count} steps (env {env_id}) to {path}\n"
                f"  Channels: {channel_summary}"
            )

    def on_pre_evaluate_policy(self) -> None:
        env = self._get_env()
        sim = env.simulator

        # Shared (robot/sim) metadata, identical across envs except env_id.
        robot_cfg = env.robot_config
        asset_cfg = robot_cfg.asset
        shared_metadata = {
            "dt": float(env.dt),
            "fps": round(1.0 / float(env.dt)),
            "sim_dt": float(env.sim_dt),
            "sim_fps": round(1.0 / float(env.sim_dt)),
            "control_decimation": env.simulator.simulator_config.sim.control_decimation_steps,
            "effort_limits": list(robot_cfg.dof_effort_limit_list),
            "dof_pos_lower_limits": list(robot_cfg.dof_pos_lower_limit_list),
            "dof_pos_upper_limits": list(robot_cfg.dof_pos_upper_limit_list),
            "velocity_limits": list(robot_cfg.dof_vel_limit_list),
            "urdf_path": str(Path(asset_cfg.asset_root) / asset_cfg.urdf_file),
        }
        if hasattr(sim, "dof_names"):
            shared_metadata["dof_names"] = list(sim.dof_names)
        if hasattr(sim, "body_names"):
            shared_metadata["body_names"] = list(sim.body_names)

        # Resolved per-DOF PD gains actually used by the sim this run (ordered to
        # match dof_names). These come from the checkpoint's saved training config
        # -- eval reloads robot.control.stiffness/damping from holosoma_config.yaml
        # -- so recording them makes each report self-document the exact gains the
        # policy was trained & simulated with (a CLI --robot.control.* override
        # would show up here too, instead of being silently invisible).
        p_gains, d_gains = self._extract_pd_gains(env)
        if p_gains is not None:
            shared_metadata["p_gains"] = p_gains
            shared_metadata["d_gains"] = d_gains

        channel_names = [
            "dof_pos_target", "dof_pos", "dof_vel", "torques", "torques_substep",
            "dof_pos_substep", "dof_vel_substep", "actions", "root_pos", "root_quat_xyzw",
            "root_lin_vel", "root_ang_vel", "body_pos_w", "body_quat_xyzw",
            "contact_forces_w", "commanded_velocity",
        ]

        # Assign each target env its frozen command (multi-task mode) and init
        # its buffers/metadata.
        commands = getattr(getattr(env, "command_manager", None), "commands", None)
        for target in self.targets:
            env_id = target["env_id"]
            self._metadata[env_id] = {**shared_metadata, "env_id": env_id}
            self._buffers[env_id] = {name: [] for name in channel_names}
            if target["command"] is not None and commands is not None:
                commands[env_id] = torch.as_tensor(
                    target["command"], device=commands.device, dtype=commands.dtype
                )
        self._assign_commands(env)  # hold commands (re-asserted each step too)

        logger.info(
            f"EvalRecordingCallback: recording {len(self.targets)} target(s): "
            + ", ".join(f"env{t['env_id']}->{Path(t['output_path']).name}" for t in self.targets)
        )

    def _assign_commands(self, env: Any) -> None:
        """Re-assert each multi-task target's frozen command on the command tensor."""
        commands = getattr(getattr(env, "command_manager", None), "commands", None)
        if commands is None:
            return
        for target in self.targets:
            if target["command"] is not None:
                commands[target["env_id"]] = torch.as_tensor(
                    target["command"], device=commands.device, dtype=commands.dtype
                )

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        env = self._get_env()
        sim = env.simulator

        # Hold each multi-task env's command (resampling is off in eval, but
        # re-asserting is cheap insurance against any reset/manager overwrite).
        self._assign_commands(env)

        def _to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().copy()

        contact_forces = getattr(sim, "contact_forces", None)
        commands = getattr(getattr(env, "command_manager", None), "commands", None)

        for target in self.targets:
            eid = target["env_id"]
            buf = self._buffers[eid]
            buf["dof_pos"].append(_to_np(sim.dof_pos[eid]))  # post_eval_env_step, after decimation
            buf["dof_vel"].append(_to_np(sim.dof_vel[eid]))
            buf["torques"].append(_to_np(self._extract_torques(env, eid)))

            # robot_root_states: [num_envs, 13] = pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)
            root = sim.robot_root_states[eid]
            buf["root_pos"].append(_to_np(root[:3]))
            buf["root_quat_xyzw"].append(_to_np(root[3:7]))
            buf["root_lin_vel"].append(_to_np(root[7:10]))
            buf["root_ang_vel"].append(_to_np(root[10:13]))

            buf["body_pos_w"].append(_to_np(sim._rigid_body_pos[eid]))
            buf["body_quat_xyzw"].append(_to_np(sim._rigid_body_rot[eid]))

            # Net contact force per body (world frame), [num_bodies, 3], ordered to
            # match body_names. Used for heel-strike gait segmentation.
            if contact_forces is not None:
                buf["contact_forces_w"].append(_to_np(contact_forces[eid]))

            # substep tensors: [decimation, num_dof] — one row per physics sub-step
            torques_substep, dof_pos_substep, dof_vel_substep = self._extract_substep_data(env, eid)
            buf["torques_substep"].append(_to_np(torques_substep))
            buf["dof_pos_substep"].append(_to_np(dof_pos_substep))
            buf["dof_vel_substep"].append(_to_np(dof_vel_substep))

            if "actions" in actor_state and actor_state["actions"] is not None:
                buf["actions"].append(_to_np(actor_state["actions"][eid]))

            buf["dof_pos_target"].append(_to_np(self._extract_dof_pos_target(env, eid)))

            if commands is not None:
                try:
                    buf["commanded_velocity"].append(_to_np(commands[eid]))
                except (IndexError,):
                    pass

        self._step_count += 1
        return actor_state

    def _extract_dof_pos_target(self, env: Any, env_id: int) -> torch.Tensor:
        """Extract desired target joint positions from the action manager's joint control term.

        The PD target is: actions_after_delay * action_scales + default_dof_pos.
        Returns shape [num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "_actions_after_delay") and hasattr(term, "action_scales"):
                return term._actions_after_delay[env_id] * term.action_scales + env.default_dof_pos[env_id]
        raise RuntimeError("No action term with _actions_after_delay found")

    def _extract_pd_gains(self, env: Any) -> tuple[list[float] | None, list[float] | None]:
        """Resolved per-DOF PD gains (p_gains, d_gains) from the joint-control term.

        The term stores them as [num_dof] tensors ordered to match dof_names (see
        JointPositionActionTerm._configure_pd_gains). Returns (None, None) if no
        such term is present (e.g. a torque-control policy), so recording stays
        best-effort.
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "p_gains") and hasattr(term, "d_gains"):
                return (
                    term.p_gains.detach().cpu().numpy().tolist(),
                    term.d_gains.detach().cpu().numpy().tolist(),
                )
        return None, None

    def _extract_torques(self, env: Any, env_id: int) -> torch.Tensor:
        """Extract torques from the action manager's joint control term.

        Returns torques, shape [num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "torques"):
                return term.torques[env_id]
        raise RuntimeError("No action term with torques found")

    def _extract_substep_data(self, env: Any, env_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract sub-step torques, dof_pos, and dof_vel from the action manager's joint control term.

        Returns (torques_substep, dof_pos_substep, dof_vel_substep), each shape [decimation, num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "torques_substep"):
                return term.torques_substep[env_id], term.dof_pos_substep[env_id], term.dof_vel_substep[env_id]
        raise RuntimeError("No action term with torques_substep found")

    def on_post_evaluate_policy(self) -> None:
        self._save()
        self._finalize_video()

    def _finalize_video(self) -> None:
        """Encode + save any in-progress eval video.

        The video recorder only encodes frames to a file on stop_recording()
        (normally fired on episode end). Evaluation runs one continuous,
        never-ending episode and eval_agent.py's shutdown path
        (close_simulation_app) does not call the recorder's cleanup, so without
        this the captured frames are discarded and no video file is written.
        Calling stop_recording() here flushes the eval video on completion.

        No-op when video recording is not active.
        """
        simulator = getattr(self._get_env(), "simulator", None)
        video_recorder = getattr(simulator, "video_recorder", None)
        if video_recorder is None or not getattr(video_recorder, "is_recording", False):
            return
        try:
            video_recorder.stop_recording()
            logger.info("EvalRecordingCallback: finalized eval video.")
        except Exception as exc:  # don't let a video error abort eval teardown
            logger.warning(f"EvalRecordingCallback: failed to finalize video: {exc}")
