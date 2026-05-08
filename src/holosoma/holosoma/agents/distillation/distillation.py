"""Pure-DAgger (behavior-cloning) distillation algorithm.

Parallels :class:`~holosoma.agents.distillation_ppo.distillation_ppo.DistillationPPO`
but strips away every PPO-only concern: no critic, no value loss, no surrogate
/ advantage computation, no action-noise std param group. The loss is just
``loss_fn(student_mean, teacher_action)`` and the optimizer is Adam over
``student + depth_backbone``.

This mirrors far-tracking's ``DepthDistillation`` (used by
``Warp-Distillation-Flat-G1-v0``) and exists as a first-class algorithm so the
FAR-pi pure-DAgger preset doesn't have to pretend-via-huge-schedule inside
``DistillationPPO``.
"""

from __future__ import annotations

import itertools
import os
from typing import Any, Callable, TypedDict

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.distillation.rollout_storage import RolloutStorageDagger
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.module_utils import setup_depth_student_teacher_module
from holosoma.agents.modules.student_teacher_modules import DepthStudentTeacher
from holosoma.config_types.algo import DistillationConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate


# Observation-group names. Kept identical to DistillationPPO's so an extension
# can swap algos without retouching obs_cfg wiring.
ACTOR_GROUP = "actor_obs"
TEACHER_GROUP = "teacher_obs"
DEPTH_GROUP = "depth_camera"
DEPTH_TERM = "depth_cam"


class DaggerMinibatch(TypedDict):
    actor_obs: torch.Tensor
    teacher_obs: torch.Tensor
    depth_obs: torch.Tensor
    actions: torch.Tensor
    teacher_actions: torch.Tensor
    dones: torch.Tensor


class Distillation(BaseAlgo):
    """Pure-DAgger trainer for a student/teacher module.

    Same public surface as :class:`DistillationPPO` (``setup``/``learn``/
    ``save``/``load``/``load_teacher``/``get_inference_policy``/
    ``evaluate_policy``) so ``train_agent.py`` can dispatch to either without
    branching. There is no critic and no PPO loss.
    """

    config: DistillationConfig
    policy: DepthStudentTeacher

    def __init__(
        self,
        env: BaseTask,
        config: DistillationConfig,
        log_dir: str | os.PathLike,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(env, config, device, multi_gpu_cfg)
        if self.is_multi_gpu:
            raise NotImplementedError("Distillation does not yet support multi-GPU training.")

        if self.config.empirical_normalization:
            raise ValueError("empirical_normalization is not supported by Distillation.")

        self.log_dir = str(log_dir)
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=self.config.num_steps_per_env,
            num_learning_iterations=self.config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

        self.current_learning_iteration = 0
        self.eval_callbacks: list[RLEvalCallback] = []
        self._warmup_active = False

        # See DistillationPPO: set by train_agent before setup() when
        # training.teacher_checkpoint is a comma-separated list.
        self.num_teachers: int = 1

        if self.config.distill_loss_type == "mse":
            self.distill_loss_fn = F.mse_loss
        elif self.config.distill_loss_type == "huber":
            self.distill_loss_fn = F.huber_loss
        else:
            raise ValueError(f"Unknown distill_loss_type: {self.config.distill_loss_type!r}")

        _ = self.env.reset_all()

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        logger.info("Setting up Distillation (pure DAgger)")
        self._resolve_obs_dims()
        self._build_policy()
        self._build_optimizer()
        self._build_storage()

    def _resolve_obs_dims(self) -> None:
        assert self.env.observation_manager is not None
        obs_dims = self.env.observation_manager.get_obs_dims()

        for grp in (ACTOR_GROUP, TEACHER_GROUP):
            if grp not in obs_dims:
                raise KeyError(f"Distillation expects observation group {grp!r} to be present.")
        if DEPTH_GROUP not in obs_dims:
            raise KeyError(f"Distillation expects observation group {DEPTH_GROUP!r} for depth input.")

        actor_dim = obs_dims[ACTOR_GROUP]
        teacher_dim = obs_dims[TEACHER_GROUP]
        if not (isinstance(actor_dim, int) and isinstance(teacher_dim, int)):
            raise TypeError(
                "actor/teacher obs groups must concatenate into a flat int dimension; "
                "got non-int entry in observation_manager.get_obs_dims()."
            )

        depth_entry = obs_dims[DEPTH_GROUP]
        if isinstance(depth_entry, dict):
            if DEPTH_TERM not in depth_entry:
                raise KeyError(
                    f"depth obs group {DEPTH_GROUP!r} missing term {DEPTH_TERM!r}; found {list(depth_entry.keys())}"
                )
            depth_tensor = self.env.observation_manager.compute_group(DEPTH_GROUP, modify_history=False)
            if not isinstance(depth_tensor, dict):
                raise TypeError(
                    f"expected compute_group({DEPTH_GROUP!r}) to return a dict when concatenate=False."
                )
            depth_shape = tuple(depth_tensor[DEPTH_TERM].shape[1:])
        elif isinstance(depth_entry, int):
            depth_shape = (depth_entry,)
        else:
            raise TypeError(f"Unexpected type for depth obs dims: {type(depth_entry)}")

        self.num_act = self.env.robot_config.actions_dim
        self.actor_obs_dim = actor_dim
        self.teacher_obs_dim = teacher_dim
        self.depth_shape = depth_shape

        logger.info(
            f"Distillation obs dims -- actor: {actor_dim}, teacher: {teacher_dim}, "
            f"depth: {depth_shape}, actions: {self.num_act}"
        )

    def _build_policy(self) -> None:
        self.policy = setup_depth_student_teacher_module(
            num_actor_obs=self.actor_obs_dim,
            num_teacher_obs=self.teacher_obs_dim,
            num_actions=self.num_act,
            module_config=self.config.module,
            device=self.device,
            num_teachers=self.num_teachers,
        )

    def _build_optimizer(self) -> None:
        """Single Adam over student MLP + depth CNN backbone. Teacher is frozen;
        ``self.policy.std`` is untrained in pure DAgger so we don't include it."""
        trainable_params = (
            list(self.policy.student.parameters())
            + list(self.policy.depth_backbone.parameters())
        )
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.config.learning_rate)
        self.learning_rate = self.config.learning_rate

    def _build_storage(self) -> None:
        self.storage = RolloutStorageDagger(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.config.num_steps_per_env,
            actor_obs_dim=self.actor_obs_dim,
            teacher_obs_dim=self.teacher_obs_dim,
            depth_shape=self.depth_shape,
            num_actions=self.num_act,
            device=self.device,
        )

    # --------------------------------------------------------------- obs helpers

    def _extract_groups(
        self, obs_dict: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pull (actor_obs, teacher_obs, depth_obs) from a manager obs dict."""
        actor_obs = obs_dict[ACTOR_GROUP]
        teacher_obs = obs_dict[TEACHER_GROUP]
        depth_entry = obs_dict[DEPTH_GROUP]
        if isinstance(depth_entry, dict):
            depth_obs = depth_entry[DEPTH_TERM]
        else:
            depth_obs = depth_entry
        return actor_obs, teacher_obs, depth_obs

    # --------------------------------------------------------------- train mode

    def _train_mode(self) -> None:
        self.policy.train()
        self.policy.teacher.eval()

    def _eval_mode(self) -> None:
        self.policy.eval()

    # -------------------------------------------------------------------- learn

    def learn(self, num_learning_iterations: int | None = None) -> None:
        self._train_mode()

        obs_dict = self.env.reset_all()
        for k in obs_dict:
            if isinstance(obs_dict[k], torch.Tensor):
                obs_dict[k] = obs_dict[k].to(self.device)
            elif isinstance(obs_dict[k], dict):
                obs_dict[k] = {kk: vv.to(self.device) for kk, vv in obs_dict[k].items()}

        total_iters = (
            num_learning_iterations if num_learning_iterations is not None else self.config.num_learning_iterations
        )
        start_iter = self.current_learning_iteration
        stop_iter = start_iter + total_iters

        for it in range(start_iter, stop_iter):
            self.current_learning_iteration = it
            self._warmup_active = (it - start_iter) < self.config.distillation_warmup_steps

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))

    # ------------------------------------------------------------- rollout step

    def _rollout_step(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                actor_obs, teacher_obs, depth_obs = self._extract_groups(obs_dict)

                # Student samples an action; teacher provides the DAgger target.
                actions = self.policy.act({"actor_obs": actor_obs, "depth_obs": depth_obs})
                teacher_actions = self.policy.teacher_act(teacher_obs).detach()

                # Warm-up: follow teacher actions so the student sees expert
                # trajectories before its policy is competent. Still record the
                # student's sampled action so the update step's distribution
                # state is consistent with what would have been taken.
                step_actions = teacher_actions if self._warmup_active else actions
                next_obs, rewards, dones, _infos = self.env.step({"actions": step_actions})
                for k in next_obs:
                    if isinstance(next_obs[k], torch.Tensor):
                        next_obs[k] = next_obs[k].to(self.device)
                    elif isinstance(next_obs[k], dict):
                        next_obs[k] = {kk: vv.to(self.device) for kk, vv in next_obs[k].items()}
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.storage.add(
                    actor_obs=actor_obs,
                    teacher_obs=teacher_obs,
                    depth_obs=depth_obs,
                    actions=actions,
                    teacher_actions=teacher_actions,
                    dones=dones.view(-1, 1),
                )
                self.policy.reset(dones)

                if self.log_dir is not None:
                    self.logging_helper.update_episode_stats(rewards, dones, _infos)

                obs_dict = next_obs
        return obs_dict

    # ------------------------------------------------------------ training step

    def _training_step(self) -> dict[str, float]:
        generator = self.storage.mini_batch_generator(
            self.config.num_mini_batches, self.config.num_learning_epochs
        )
        loss_accum = 0.0
        num_updates = 0

        mini_batch: DaggerMinibatch
        for mini_batch in generator:
            actor_obs = mini_batch["actor_obs"]
            depth_obs = mini_batch["depth_obs"]
            teacher_actions = mini_batch["teacher_actions"]

            # Deterministic student forward (distribution mean); this is what
            # the reference implementation uses for behavior cloning, not a
            # sampled action. Matches far-tracking's
            # DepthDistillation.update() (my_distillation.py:206-209):
            # simple MSE, no expert-terminate mask.
            student_mean = self.policy.student(
                self.policy._student_input(actor_obs, depth_obs)
            )
            loss = self.distill_loss_fn(student_mean, teacher_actions)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            loss_accum += loss.detach().item()
            num_updates += 1

        loss_dict: dict[str, float] = {"warmup_active": float(self._warmup_active)}
        if num_updates > 0:
            # Logged as "behavior" to match far-tracking's DepthDistillation
            # loss-dict key (my_distillation.py:238), keeps wandb plot axes
            # consistent across the two codebases.
            loss_dict["behavior"] = loss_accum / num_updates

        self.storage.clear()
        return loss_dict

    # ------------------------------------------------------------------- logging

    def _post_epoch_logging(self, it: int, loss_dict: dict[str, float]) -> None:
        extra_log_dicts = {
            "Policy": {
                "mean_noise_std": self.policy.std.mean().item(),
            },
            "LR": {
                "student_backbone_lr": self.learning_rate,
            },
        }
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    # ---------------------------------------------------------- save/load/teacher

    def save(self, path: str | None = None, name: str = "last.ckpt") -> None:
        if path is None:
            path = os.path.join(self.log_dir, name)
        checkpoint_dict: dict[str, Any] = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "learning_rate": self.learning_rate,
        }
        checkpoint_dict.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)

    def load(self, path: str | None) -> dict | None:
        if path is None:
            return None
        logger.info(f"Loading Distillation checkpoint from {path}")
        loaded = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(loaded["model_state_dict"])
        if "optimizer_state_dict" in loaded:
            try:
                self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
            except Exception as exc:
                logger.warning(f"Could not load optimizer state: {exc}")
        self.current_learning_iteration = int(loaded.get("iter", 0))
        if "learning_rate" in loaded:
            self.learning_rate = float(loaded["learning_rate"])
        self._restore_env_state(loaded.get("env_state"))
        self.policy.teacher.eval()
        for p in self.policy.teacher.parameters():
            p.requires_grad_(False)
        return loaded.get("infos")

    def load_teacher(self, paths: str | list[str]) -> None:
        """Load one or more teacher-only checkpoints into ``policy.teachers``.

        Same semantics as :meth:`DistillationPPO.load_teacher`; delegates to
        :meth:`StudentTeacher.load_teacher_state_dict` once per slot.
        """
        if isinstance(paths, str):
            paths = [paths]
        if len(paths) != self.policy.num_teachers:
            raise RuntimeError(
                f"load_teacher: got {len(paths)} paths but policy has "
                f"{self.policy.num_teachers} teacher slots."
            )
        for i, path in enumerate(paths):
            logger.info(f"Loading teacher[{i}] weights from {path}")
            loaded = torch.load(path, map_location=self.device)
            if isinstance(loaded, dict) and "actor_model_state_dict" in loaded:
                teacher_ckpt = {"model_state_dict": loaded["actor_model_state_dict"]}
            else:
                teacher_ckpt = loaded
            self.policy.load_teacher_state_dict(teacher_ckpt, strict=True, teacher_index=i)
        for t in self.policy.teachers:
            t.eval()
            for p in t.parameters():
                p.requires_grad_(False)

    # --------------------------------------------------------- inference / eval

    @property
    def inference_model(self) -> DepthStudentTeacher:
        return self.policy

    @property
    def actor_onnx_wrapper(self):
        raise NotImplementedError("ONNX export for Distillation will be wired up in a later phase.")

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        self.policy.eval()
        if device is not None:
            self.policy.to(device)

        @torch.no_grad()
        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.policy.act_inference(
                {
                    "actor_obs": obs["actor_obs"],
                    "depth_obs": obs.get("depth_obs"),
                }
            )

        return policy_fn

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None) -> None:
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state: dict[str, Any] = {"done_indices": [], "stop": False}
        eval_policy = self.get_inference_policy()

        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})

        actor_obs, _teacher_obs, depth_obs = self._extract_groups(obs_dict)
        actor_state["obs"]["actor_obs"] = actor_obs
        actor_state["obs"]["depth_obs"] = depth_obs

        for step in itertools.islice(itertools.count(), max_eval_steps):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state, eval_policy)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)

        self._post_evaluate_policy()

    def _create_eval_callbacks(self) -> None:
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env: bool = True) -> None:
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()
        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self) -> None:
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict, eval_policy: Callable) -> dict:
        actor_obs, _teacher_obs, depth_obs = self._extract_groups(actor_state["obs"])
        actions = eval_policy({"actor_obs": actor_obs, "depth_obs": depth_obs})
        actor_state["actions"] = actions
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state
