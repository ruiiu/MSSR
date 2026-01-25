# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .core_algos import AdvantageEstimator, FixedKLController, KLController, compute_kl, get_kl_controller
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0, 
                     spo_value_tracker=None, spo_config=None):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        # Extract entropy shaping parameters for REINFORCE++
        use_entropy_shaping = spo_config.use_entropy_shaping if (spo_config and hasattr(spo_config, 'use_entropy_shaping')) else False
        entropy_alpha = spo_config.entropy_alpha if (spo_config and hasattr(spo_config, 'entropy_alpha')) else 0.4
        entropy_kappa = spo_config.entropy_kappa if (spo_config and hasattr(spo_config, 'entropy_kappa')) else 2.0
        log_probs_for_entropy = data.batch.get("old_log_probs", None) if use_entropy_shaping else None
        
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma,
            use_entropy_shaping=use_entropy_shaping,
            log_probs=log_probs_for_entropy,
            entropy_alpha=entropy_alpha,
            entropy_kappa=entropy_kappa
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.SPO:
        # SPO uses per-sample value tracking with global normalization
        # Each sample (text + image) is tracked separately using sample_id
        
        # Get sample IDs for per-sample tracking
        # sample_id is the dataset index, uniquely identifying each (text, image) pair
        if "sample_id" in data.non_tensor_batch:
            sample_id_data = data.non_tensor_batch["sample_id"]
            # Convert to list, handling both numpy arrays and tensors
            if hasattr(sample_id_data, 'tolist'):
                sample_ids = sample_id_data.tolist()
            else:
                sample_ids = list(sample_id_data)
        else:
            # Fallback: use uid (but this won't match initialization)
            sample_ids = list(data.non_tensor_batch["uid"])
        
        # Extract KL divergences if available for adaptive forgetting
        kl_divergences = None
        old_log_probs = None
        if "old_log_probs" in data.batch and "ref_log_probs" in data.batch:
            # For per-prompt rho calculation, we need old_log_probs (current policy)
            old_log_probs = data.batch["old_log_probs"]
            
            # Compute KL divergence for adaptation (only if not using per-prompt rho)
            if not spo_value_tracker.use_per_sample_rho:
                kl_divergences = core_algos.compute_kl(
                    data.batch["old_log_probs"], 
                    data.batch["ref_log_probs"],
                    kl_penalty=spo_config.kl_penalty if spo_config else "kl"
                )
        
        # SPO uses raw task scores (no KL penalty) for both advantages and value tracking
        # Ensure use_kl_loss: true in config so KL is handled as separate loss term
        token_level_scores = data.batch.get("token_level_scores", token_level_rewards)
        
        # Extract entropy shaping parameters for SPO
        use_entropy_shaping = spo_config.use_entropy_shaping if (spo_config and hasattr(spo_config, 'use_entropy_shaping')) else False
        entropy_alpha = spo_config.entropy_alpha if (spo_config and hasattr(spo_config, 'entropy_alpha')) else 0.4
        entropy_kappa = spo_config.entropy_kappa if (spo_config and hasattr(spo_config, 'entropy_kappa')) else 2.0
        log_probs_for_entropy = old_log_probs if use_entropy_shaping else None
        
        # Compute SPO advantages with optional entropy shaping
        result = core_algos.compute_spo_advantage(
            token_level_scores=token_level_scores,  # Raw task scores (no KL penalty)
            response_mask=response_mask,
            value_tracker=spo_value_tracker,
            sample_ids=sample_ids,
            kl_divergences=kl_divergences,
            old_log_probs=old_log_probs,  # For per-prompt rho calculation
            eps=spo_config.spo_eps if spo_config else 1e-6,
            normalize_globally=spo_config.spo_normalize_globally if spo_config else True,
            use_entropy_shaping=use_entropy_shaping,
            log_probs=log_probs_for_entropy,
            entropy_alpha=entropy_alpha,
            entropy_kappa=entropy_kappa
        )
        
        # Handle return values (with or without metrics)
        if len(result) == 3:
            advantages, returns, metrics = result
        else:
            advantages, returns = result

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        # Initialize SPO value tracker and prioritized sampler if using SPO
        if config.algorithm.adv_estimator == AdvantageEstimator.SPO:
            from .core_algos import SPOValueTracker, SPOPrioritizedSampler
            # Per-prompt value tracker with Bayesian updates and KL-adaptive forgetting
            # Works for both text-only and multimodal scenarios
            self.spo_value_tracker = SPOValueTracker(
                rho_min=config.algorithm.spo_rho_min,
                rho_max=config.algorithm.spo_rho_max,
                target_kl=config.algorithm.spo_target_kl,
                v_init=config.algorithm.spo_v_init,
                use_per_sample_rho=config.algorithm.spo_per_sample_rho,
                d_half=config.algorithm.spo_d_half,
                use_fixed_rho=config.algorithm.spo_use_fixed_rho
            )
            # Prioritized sampler for adaptive curriculum learning
            # Only initialize if uncertainty weighting is enabled
            if config.algorithm.spo_use_uncertainty_weighting:
                self.spo_prioritized_sampler = SPOPrioritizedSampler(
                    use_uncertainty_weighting=True,
                    priority_alpha=config.algorithm.spo_priority_alpha,
                    priority_epsilon=config.algorithm.spo_priority_epsilon
                )
            else:
                self.spo_prioritized_sampler = None
        else:
            self.spo_value_tracker = None
            self.spo_prioritized_sampler = None

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    
    def _validate(self) -> Dict[str, Any]:
        if self.config.trainer.enable_passk_validation:
            return self._validate_passk()
        else:
            return self._validate_pass1()
    
    def _validate_pass1(self) -> Dict[str, Any]:
        """Standard pass@1 validation (single sample per problem)"""
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        
        # Rename accuracy_reward to pass@1 for clarity
        if "val/accuracy_reward" in val_reward_metrics:
            val_reward_metrics["val/pass@1"] = val_reward_metrics.pop("val/accuracy_reward")
        
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics}
    
    def _validate_passk(self) -> Dict[str, Any]:
        """Pass@k validation (k samples per problem, slower but more comprehensive)"""
        
        print(f"Start pass@k validation (slower but more comprehensive)...")
        
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        problem_results = defaultdict(list)  # problem_id -> list of accuracies
        
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        
        for batch_idx, batch_dict in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            
            # Use pass@k: generate 8 samples per problem (for pass@1, pass@4, pass@8)
            repeat_times = 8
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config.copy()
            test_gen_batch.meta_info["n"] = repeat_times  # Override to generate k samples
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # Group results by problem (8 samples per original problem)
            batch_size = test_batch.batch["prompts"].shape[0]
            n_problems = batch_size // 8
            
            for i in range(n_problems):
                problem_start = i * 8
                problem_end = (i + 1) * 8
                
                # Extract accuracies for this problem's 8 samples
                problem_accuracies = []
                for j in range(problem_start, problem_end):
                    # Get accuracy from reward metrics (assuming accuracy exists)
                    accuracy = reward_metrics.get("accuracy", [0.0] * batch_size)[j]
                    problem_accuracies.append(accuracy)
                
                problem_id = batch_idx * n_problems + i
                problem_results[problem_id] = problem_accuracies

            # store generations for logging (all samples)
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        
        # Compute pass@k metrics for multiple k values
        pass_at_1_scores = []
        pass_at_4_scores = []
        pass_at_8_scores = []
        # pass_at_k_scores = []  # Original k value
        
        for problem_id, accuracies in problem_results.items():
            # Pass@1: first sample is correct
            pass_at_1 = accuracies[0] if accuracies else 0.0
            pass_at_1_scores.append(pass_at_1)
            
            # Pass@4: at least one of first 4 samples is correct
            pass_at_4 = 1.0 if any(acc > 0.5 for acc in accuracies[:4]) else 0.0
            pass_at_4_scores.append(pass_at_4)
            
            # Pass@8: at least one of first 8 samples is correct
            pass_at_8 = 1.0 if any(acc > 0.5 for acc in accuracies[:8]) else 0.0
            pass_at_8_scores.append(pass_at_8)
            
            # # Pass@k: at least one of k samples is correct (original k value)
            # pass_at_k = 1.0 if any(acc > 0.5 for acc in accuracies) else 0.0
            # pass_at_k_scores.append(pass_at_k)
        
        # Compute overall metrics
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        
        # Make val_reward_score consistent with Pass@1 validation (weighted combination)
        if "val/overall_reward" in val_reward_metrics:
            self.val_reward_score = val_reward_metrics["val/overall_reward"]
        else:
            # Fallback to pass@1 if overall_reward not available
            self.val_reward_score = np.mean(pass_at_1_scores)
        
        # Add pass@k specific metrics for multiple k values
        val_reward_metrics["val/pass@1"] = np.mean(pass_at_1_scores)
        val_reward_metrics["val/pass@4"] = np.mean(pass_at_4_scores)
        val_reward_metrics["val/pass@8"] = np.mean(pass_at_8_scores)
        
        # Remove accuracy_reward to avoid confusion (replaced by pass@1)
        if "val/accuracy_reward" in val_reward_metrics:
            val_reward_metrics.pop("val/accuracy_reward")
        
        print(f"Finish pass@k validation.")
        print(f"Pass@1: {val_reward_metrics['val/pass@1']:.4f}, Pass@4: {val_reward_metrics['val/pass@4']:.4f}, Pass@8: {val_reward_metrics['val/pass@8']:.4f}")
        
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics}

    # def _validate(self) -> Dict[str, Any]:
    #     reward_tensor_lst = []
    #     # Lists to collect samples for the table
    #     sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
    #     reward_metrics_lst = defaultdict(list)
    #     print("Start validation...")
    #     self.actor_rollout_ref_wg.prepare_rollout_engine()
    #     for batch_dict in self.val_dataloader:
    #         test_batch = DataProto.from_single_dict(batch_dict)
    #         test_gen_batch = test_batch.pop(
    #             batch_keys=["input_ids", "attention_mask", "position_ids"],
    #             non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
    #         )
    #         repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
    #         test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
    #         test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
    #         test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
    #         test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

    #         test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
    #         test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
    #         test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

    #         # repeat to align with repeated responses in rollout
    #         test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
    #         test_batch = test_batch.union(test_output_gen_batch)

    #         # evaluate using reward_function
    #         reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

    #         # store generations
    #         input_ids = test_batch.batch["prompts"]
    #         input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
    #         output_ids = test_batch.batch["responses"]
    #         output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    #         scores = reward_tensor.sum(-1).cpu().tolist()
    #         sample_inputs.extend(input_texts)
    #         sample_outputs.extend(output_texts)
    #         sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
    #         sample_scores.extend(scores)

    #         reward_tensor_lst.append(reward_tensor)
    #         for key, value in reward_metrics.items():
    #             reward_metrics_lst[key].extend(value)

    #     self.actor_rollout_ref_wg.release_rollout_engine()
    #     self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
    #     self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
    #     val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
    #     print("Finish validation.")
    #     return {"val/reward_score": self.val_reward_score, **val_reward_metrics}

    def _create_text_only_batch(self, batch: DataProto) -> DataProto:
        """
        Create a text-only version of the batch by replacing images with blank/zero images.
        
        This is used for computing visual influence weights in visual-aware SPO.
        The text-only batch uses blank images (zeros) so the model processes "no visual information"
        while preserving sequence structure and token alignment.
        
        Why blank images instead of removing them:
        - Preserves sequence length (no need to remap positions)
        - Maintains token alignment between full and text-only outputs
        - Model can process normally without special handling
        - Direct logprob comparison is valid
        
        Args:
            batch: DataProto with full multimodal data
            
        Returns:
            text_only_batch: DataProto with blank images instead of real images
        """
        from copy import deepcopy
        
        # Create a deep copy of the batch to avoid modifying the original
        text_only_batch = deepcopy(batch)
        
        # Replace multimodal data with blank versions
        if "multi_modal_data" in text_only_batch.non_tensor_batch:
            # Get the original multi_modal_data
            multi_modal_data = text_only_batch.non_tensor_batch["multi_modal_data"]
            
            # Create blank versions for each sample
            blank_multi_modal_data = []
            for i, mm_data in enumerate(multi_modal_data):
                if mm_data is not None and isinstance(mm_data, dict):
                    # Create blank version with same structure but modified visual data
                    blank_mm_data = {}
                    
                    # Copy all keys first
                    for key, value in mm_data.items():
                        blank_mm_data[key] = value
                    
                    # Replace visual data with blank versions
                    if "images" in mm_data:
                        # Replace images with empty list to create text-only baseline
                        blank_mm_data["images"] = []
                    
                    blank_multi_modal_data.append(blank_mm_data)
                else:
                    # No multimodal data for this sample
                    blank_multi_modal_data.append(None)
            
            # Replace with blank data
            text_only_batch.non_tensor_batch["multi_modal_data"] = np.array(
                blank_multi_modal_data, dtype=object
            )
        
        return text_only_batch
    
    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: Dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            
            # Add sample_id for SPO per-sample tracking
            # sample_id comes from dataset and uniquely identifies each (text, image) combination
            # This is the dataset index, which is stable and unique
            if "sample_id" in batch_dict:
                # sample_id is already in the batch from dataset
                new_batch.non_tensor_batch["sample_id"] = batch_dict["sample_id"]
            else:
                # Fallback: use uid as sample_id (for datasets without sample_id)
                new_batch.non_tensor_batch["sample_id"] = new_batch.non_tensor_batch["uid"]
            
            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise ValueError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def _initialize_spo_from_dataset(self):
        """
        Initialize SPO value tracker by running policy n_init times through entire dataset.
        
        This is a simplified alternative to multi-step collection:
        - Iterate through full dataset n_init times
        - Generate 1 response per prompt per iteration
        - Use average of n_init outcomes as initial value estimate
        - Every prompt gets initialized (no fallback needed)
        """
        n_init = self.config.algorithm.spo_n_init
        print("\n" + "="*80)
        print("SPO Algorithm Initialization")
        print("="*80)
        print(f"Running policy {n_init} time(s) through entire dataset to collect initial value estimates...")
        print(f"Dataset size: {len(self.train_dataloader.dataset)} prompts\n")
        
        from collections import defaultdict
        dataset_idx_to_outcomes = defaultdict(list)  # Maps dataset_index -> list of outcomes
        total_samples_processed = 0
        
        # Create a temporary dataloader for initialization:
        # - No shuffling (deterministic)
        # - No drop_last (process all prompts)
        # - Smaller batch size if needed for memory
        from torch.utils.data import DataLoader
        from verl.utils.dataset import collate_fn
        
        init_batch_size = 4096 #min(512, len(self.train_dataloader.dataset))
        init_dataloader = DataLoader(
            self.train_dataloader.dataset,
            batch_size=init_batch_size,
            shuffle=False,  # Deterministic initialization
            drop_last=True,  # Process ALL prompts
            collate_fn=collate_fn,
            num_workers=0,  # Simple single-process loading
        )
        
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        
        # Run through dataset n_init times
        for init_round in range(n_init):
            print(f"Initialization round {init_round + 1}/{n_init}...")
            current_dataset_idx = 0  # Track which dataset sample we're processing
            
            for batch_idx, batch_dict in enumerate(init_dataloader):
                # Convert to DataProto
                meta_info = {
                    "min_pixels": self.config.data.min_pixels,
                    "max_pixels": self.config.data.max_pixels,
                    "video_fps": self.config.data.video_fps,
                }
                batch_data = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
                
                # Extract generation batch
                gen_batch = batch_data.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
                )
                
                # Generate responses (n=1 for initialization)
                gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
                
                # Union with batch_data to get full batch
                full_batch = batch_data.union(gen_batch_output)
                
                # Get sample IDs from dataset - these uniquely identify each (text, image) pair
                # sample_id is the dataset index, stable across runs
                if "sample_id" in batch_dict:
                    sample_ids = batch_dict["sample_id"].tolist() if hasattr(batch_dict["sample_id"], 'tolist') else list(batch_dict["sample_id"])
                else:
                    # Fallback: use current dataset index
                    sample_ids = [current_dataset_idx + i for i in range(len(full_batch.batch))]
                
                # Compute rewards
                reward_tensor, reward_metrics = ray.get(
                    self.reward_fn.compute_reward.remote(full_batch)
                )
                
                # Store outcome for each dataset sample
                # sample_id uniquely identifies each (text, image) pair
                batch_size = len(full_batch.batch)
                for i in range(batch_size):
                    # Get the sequence-level binary outcome {0, 1}
                    # For SPO with Beta-Bernoulli, we use pure binary rewards
                    # The value tracker will use this to estimate V(x) = P(correct | x)
                    if len(reward_tensor.shape) > 1:
                        # Token-level rewards: sum to get sequence score
                        sequence_score = reward_tensor[i].sum().item()
                    else:
                        # Sequence-level reward already
                        sequence_score = reward_tensor[i].item()
                    
                    # Append the binary outcome (0.0 or 1.0) to the list for this sample
                    # For math task: pure accuracy reward (correct=1, incorrect=0)
                    sample_id = sample_ids[i]
                    dataset_idx_to_outcomes[sample_id].append(sequence_score)
                    total_samples_processed += 1
                
                current_dataset_idx += batch_size
        
        self.actor_rollout_ref_wg.release_rollout_engine()
        
        # Initialize value tracker with dataset indices
        dataset_indices = list(dataset_idx_to_outcomes.keys())
        outcomes = [dataset_idx_to_outcomes[idx] for idx in dataset_indices]  # List of lists (each sample has n_init outcomes)
        
        print(f"\n{'='*80}")
        print("SPO Algorithm Initialization Complete")
        print("="*80)
        print(f"Initialized {len(dataset_indices)} dataset samples (from {total_samples_processed} processed)")
        print(f"Each sample evaluated {n_init} time(s) for better initial value estimates")
        print(f"Each sample ID uniquely identifies a (text, image) combination from the dataset")
        
        if len(dataset_indices) > 0:
            self.spo_value_tracker.initialize_from_samples(dataset_indices, outcomes)
            
            # Show success rate statistics (average across all outcomes)
            all_outcomes = [outcome for outcomes_list in outcomes for outcome in outcomes_list]
            print(f"Initial success rate: mean={np.mean(all_outcomes):.3f}, std={np.std(all_outcomes):.3f}")
            print(f"Average outcomes per sample: {total_samples_processed / len(dataset_indices):.1f}")
            print(f"\nDuring training: sample_id from dataset will match to initialized values")
        else:
            print("Warning: No samples collected. Using fallback initialization.")
        
        print("="*80 + "\n")
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # Initialize SPO value tracker (if using SPO and not resuming from checkpoint)
        if (self.config.algorithm.adv_estimator == AdvantageEstimator.SPO and 
            self.global_step == 0 and 
            self.config.algorithm.spo_run_initialization):
            # Run policy once through dataset to initialize all prompts
            self._initialize_spo_from_dataset()
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.SPO and self.global_step == 0:
            print("\n" + "="*80)
            print("SPO Initialization: SKIPPED (spo_run_initialization=False)")
            print(f"Using fallback initialization with v_init={self.config.algorithm.spo_v_init} for all prompts")
            print("="*80 + "\n")

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)
                    
                    # Compute text-only KL divergence for SPO (if enabled)
                    # Text-only stream is used to regularize multimodal policy
                    if (self.config.algorithm.adv_estimator == AdvantageEstimator.SPO and 
                        hasattr(self.config.algorithm, 'text_kl_enabled') and
                        self.config.algorithm.text_kl_enabled):
                        
                        # Check if we should use text KL this step (annealing)
                        use_text_kl = True
                        text_kl_annealing_prob = 1.0
                        
                        if (hasattr(self.config.algorithm, 'text_kl_annealing') and 
                            self.config.algorithm.text_kl_annealing):
                            # Compute current annealing probability
                            current_step = self.global_step
                            start_step = self.config.algorithm.text_kl_annealing_start_step
                            end_step = self.config.trainer.max_steps  # Use trainer.max_steps as end step
                            start_prob = self.config.algorithm.text_kl_annealing_start_prob
                            end_prob = self.config.algorithm.text_kl_annealing_end_prob
                            
                            if current_step < start_step:
                                text_kl_annealing_prob = start_prob
                            elif current_step >= end_step:
                                text_kl_annealing_prob = end_prob
                            else:
                                # Linear interpolation
                                progress = (current_step - start_step) / (end_step - start_step)
                                text_kl_annealing_prob = start_prob + (end_prob - start_prob) * progress
                            
                            # Decide whether to use text KL this step
                            import random
                            use_text_kl = random.random() < text_kl_annealing_prob
                            
                            # Log annealing probability
                            metrics["text_kl/annealing_prob"] = text_kl_annealing_prob
                            metrics["text_kl/use_text_kl"] = float(use_text_kl)
                        
                        if use_text_kl:
                            with timer("text_kl_computation", timing_raw):
                                # Compute text-only logprobs (with blank images)
                                text_only_batch = self._create_text_only_batch(batch)
                                
                                # Force cache invalidation by temporarily changing uid
                                # The FSDPWorker caches multi_modal_inputs based on uid, but we're using
                                # the same batch with different visual inputs (real vs blank images)
                                original_uids = text_only_batch.non_tensor_batch["uid"].copy()
                                
                                # Create new unique UIDs to force cache miss
                                import uuid
                                new_uids = np.array([str(uuid.uuid4()) for _ in range(len(original_uids))], dtype=object)
                                text_only_batch.non_tensor_batch["uid"] = new_uids
                                
                                text_only_log_probs = self.actor_rollout_ref_wg.compute_log_probs(text_only_batch)
                                
                                # Restore original UIDs to avoid downstream issues
                                text_only_batch.non_tensor_batch["uid"] = original_uids
                                
                                # Extract full modality logprobs (already computed)
                                full_log_probs = batch.batch["old_log_probs"]
                                text_only_logprobs_tensor = text_only_log_probs.batch["old_log_probs"]
                                
                                # Ensure we have plain tensors, not TensorDict objects
                                if hasattr(full_log_probs, 'batch'):
                                    full_log_probs = full_log_probs.batch.get("old_log_probs", full_log_probs)
                                if hasattr(text_only_logprobs_tensor, 'batch'):
                                    text_only_logprobs_tensor = text_only_logprobs_tensor.batch.get("old_log_probs", text_only_logprobs_tensor)
                                
                                # Compute KL divergence between text-only and multimodal streams
                                text_kl_divergence, text_kl_metrics = core_algos.compute_kl_divergence_between_streams(
                                    multimodal_log_probs=full_log_probs,
                                    text_only_log_probs=text_only_logprobs_tensor,
                                    response_mask=batch.batch["response_mask"]
                                )
                                # Store KL divergence in batch for actor update
                                batch.batch["text_kl_divergence"] = text_kl_divergence
                                # Log KL metrics
                                metrics.update(text_kl_metrics)
                                metrics["text_kl/annealing_skipped"] = 0.0
                        else:
                            # Skip text KL this step (single-stream)
                            metrics["text_kl/annealing_skipped"] = 1.0
                    
                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        spo_value_tracker=self.spo_value_tracker,
                        spo_config=self.config.algorithm
                    )
                    
                    # Update SPO prioritized sampler weights after advantage computation
                    if (self.config.algorithm.adv_estimator == AdvantageEstimator.SPO and 
                        self.spo_prioritized_sampler is not None):
                        # Update uncertainty-based weights from value tracker
                        if self.spo_prioritized_sampler.use_uncertainty_weighting:
                            # Get all sample IDs that have been initialized
                            all_sample_ids = list(self.spo_value_tracker.prompt_alpha.keys())
                            if len(all_sample_ids) > 0:
                                self.spo_prioritized_sampler.update_weights_from_value_tracker(
                                    self.spo_value_tracker, all_sample_ids
                                )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
            
            # Add SPO beta distribution statistics for logging
            if self.config.algorithm.adv_estimator == AdvantageEstimator.SPO and self.spo_value_tracker is not None:
                beta_stats = self.spo_value_tracker.get_beta_distribution_stats()
                metrics.update(beta_stats)

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
