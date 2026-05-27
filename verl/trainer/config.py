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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        if self.image_dir is not None:
            if os.path.exists(self.image_dir):  # ray job uses absolute path
                self.image_dir = os.path.abspath(self.image_dir)
            else:
                print(f"Image directory {self.image_dir} not found.")
                self.image_dir = None

        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):  # ray job uses absolute path
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                print(f"Format prompt file {self.format_prompt} not found.")
                self.format_prompt = None


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`, `mvsr`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    
    # MVSR-specific configuration parameters for the vanilla single-rollout baseline
    mvsr_rho_min: float = 0.875
    """minimum forgetting rate for MVSR (corresponds to W_max=25)"""
    mvsr_rho_max: float = 0.96
    """maximum forgetting rate for MVSR (corresponds to W_min=8)"""
    mvsr_target_kl: float = 0.06
    """target KL divergence for adaptive forgetting in MVSR (used with global adaptive rho)"""
    mvsr_run_initialization: bool = True
    """whether to run Algorithm initialization (run policy once through dataset). If False, uses fallback v_init for all prompts."""
    mvsr_n_init: int = 1
    """number of times to sample each prompt during initialization to get better initial value estimates"""
    mvsr_v_init: float = 0.5
    """fallback value for prompts not in initialization set, or for all prompts if mvsr_run_initialization=False"""
    mvsr_normalize_globally: bool = True
    """whether to normalize advantages globally across the batch in MVSR (critical feature)"""
    mvsr_eps: float = 1e-6
    """epsilon for numerical stability in MVSR"""
    
    # Per-prompt rho calculation (project baseline style)
    mvsr_per_sample_rho: bool = False
    """use per-sample rho calculation (True) or global adaptive rho (False).
    Per-sample: each sample gets its own rho by comparing log probs when it reappears.
    Global: all samples use same rho based on global KL history."""
    mvsr_d_half: float = 0.06
    """half-life parameter for exponential decay in per-prompt rho: ρ = 2^(-D/D_half)"""
    
    # Note: MVSR uses per-prompt value tracking with Bayesian updates
    # Works for both text-only and multimodal scenarios
    
    # MVSR Weighted Curriculum Sampling (project baseline style)
    mvsr_use_uncertainty_weighting: bool = True
    """enable uncertainty-based weighted curriculum sampling: weight = sqrt(p*(1-p))"""
    mvsr_priority_alpha: float = 1.0
    """scaling factor for uncertainty weights (1.0=linear, >1=emphasize high uncertainty)"""
    mvsr_priority_epsilon: float = 1e-6
    """epsilon for MVSR prioritized sampling to avoid zero weights"""

    mvsr_kl_window_size: int = 20
    """window size for computing average KL in global adaptive rho calculation"""
    
    # Entropy-based advantage shaping (applicable to all algorithms)
    use_entropy_shaping: bool = False
    """enable entropy-based advantage shaping for all algorithms (PPO, GRPO, RLOO, MVSR, etc.).
    This encourages exploration by adding an entropy-based term to advantages.
    The formula is: ψ(H_t) = min(α·H_t^detach, |A_t|/κ)
    Can be used with any advantage estimator."""
    entropy_alpha: float = 0.4
    """scaling factor for entropy term in advantage shaping.
    Higher values (e.g., 0.5-1.0) encourage more exploration,
    lower values (e.g., 0.1-0.3) provide subtle exploration signal.
    Recommended range: 0.2-0.6."""
    entropy_kappa: float = 2.0
    """denominator for advantage magnitude term in entropy shaping.
    Controls the maximum entropy bonus relative to advantage magnitude.
    Lower values (e.g., 1.0-1.5) allow larger entropy bonuses,
    higher values (e.g., 2.0-4.0) cap the entropy bonus more conservatively.
    Recommended: 2.0."""
    
    # Entropy loss regularization for encouraging exploration
    use_entropy_loss: bool = False
    """enable entropy loss regularization to encourage exploration.
    This adds negative entropy to the policy gradient loss (maximize entropy = minimize -H).
    Different from entropy shaping which modifies advantages.
    Can be used with or without MVSR, PPO, or other algorithms."""
    entropy_coef: float = 0.01
    """coefficient for entropy loss regularization.
    Higher values (e.g., 0.05-0.1) provide stronger exploration bonus,
    lower values (e.g., 0.001-0.01) provide subtle exploration signal.
    Recommended range: 0.001-0.05 depending on task."""
    
    # Text-only KL divergence regularization (between text-only and multimodal streams)
    text_kl_enabled: bool = False
    """enable KL divergence regularization between text-only and multimodal streams.
    When enabled, creates a text-only stream (with blank images) to compute
    KL(text_only || multimodal) where text-only is the reference/anchor policy.
    This regularizes the multimodal policy to not diverge too far from text-only baseline,
    preventing overfitting to visual features while allowing helpful visual grounding."""
    text_kl_coef: float = 0.01
    """coefficient for text-only KL divergence regularization.
    Higher values (e.g., 0.5-1.0) provide stronger regularization,
    lower values (e.g., 0.01-0.1) provide subtle regularization.
    Recommended range: 0.05-0.3 depending on how much you want to constrain
    the multimodal policy to stay close to text-only policy."""
    
    # Text KL annealing: gradually phase out text-only KL regularization during training
    text_kl_annealing: bool = False
    """enable annealing for text-only KL regularization. When enabled, text KL probability
    starts high and gradually decreases to zero over training, allowing model to learn
    with regularization early then train freely later."""
    text_kl_annealing_start_prob: float = 1.0
    """initial probability of using text KL regularization (1.0 = always use, 0.0 = never use).
    Typically set to 1.0 to start with full text KL regularization."""
    text_kl_annealing_end_prob: float = 0.0
    """final probability of using text KL regularization after annealing.
    Typically set to 0.0 to phase out text KL regularization completely."""
    text_kl_annealing_start_step: int = 0
    """training step to start annealing (0 = from beginning).
    Annealing ends at trainer.max_steps (no separate end_step needed)."""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # ray job uses absolute path
        if self.load_checkpoint_path is not None:
            if os.path.exists(self.load_checkpoint_path):  # ray job uses absolute path
                self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)
            else:
                print(f"Model checkpoint {self.load_checkpoint_path} not found.")
                self.load_checkpoint_path = None


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        self.worker.actor.use_entropy_loss = self.algorithm.use_entropy_loss
        self.worker.actor.entropy_coef = self.algorithm.entropy_coef
        self.worker.actor.text_kl_enabled = self.algorithm.text_kl_enabled
        self.worker.actor.text_kl_coef = self.algorithm.text_kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
