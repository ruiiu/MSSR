# MSSR

<!-- This repository is an [EasyR1](https://github.com/hiyouga/EasyR1)/[veRL](https://github.com/volcengine/verl) fork for single-rollout multimodal RL on visual math reasoning. -->

The terminology in this folder is:

- **MVSR**: Multimodal Vanilla Single-Rollout. This is the vanilla baseline: one rollout per prompt, prompt-level value tracking, and global advantage normalization.
- **MSSR**: Multimodal Stabilized Single-Rollout. This is our approach: MVSR plus the stabilization recipe used by the main MSSR scripts, especially entropy-based advantage shaping.

## What This Approach Does

GRPO samples multiple responses per prompt and uses within-group relative rewards. MVSR/MSSR instead uses one response per prompt:

1. Generate one rollout for each multimodal prompt with `worker.rollout.n=1`.
2. Use a persistent prompt-level value tracker as the baseline.
3. Compute the advantage as the outcome score minus the tracked prompt baseline.
4. Normalize advantages globally across the batch for stable policy updates.
5. For MSSR, add entropy-based advantage shaping and the stabilized launch settings used by `examples/7b_mssr.sh` and `examples/3b_mssr.sh`.

MVSR is the baseline. MSSR is the stabilized method built on top of it.

## Main Code Paths

| File | Purpose |
| --- | --- |
| `verl/trainer/core_algos.py` | MVSR value tracker, prioritized sampler, and advantage computation. |
| `verl/trainer/ray_trainer.py` | MVSR initialization, value updates, entropy shaping, and optional text-only KL. |
| `verl/trainer/config.py` | `mvsr_*` config fields and MSSR stabilization knobs. |
| `examples/config_mssr.yaml` | Main MSSR config. |
| `examples/reward_function/math.py` | Math reward for visual reasoning scripts. |
<!-- | `tests/test_mvsr.py` | MVSR unit tests. | -->

## Scripts

| Variant | Script | Description |
| --- | --- | --- |
| MSSR | `examples/7b_mssr.sh` | Qwen2.5-VL-7B stabilized single-rollout method. |
| MSSR | `examples/3b_mssr.sh` | Qwen2.5-VL-3B stabilized single-rollout method. |
| MVSR | `examples/7b_mvsr.sh` | Qwen2.5-VL-7B vanilla single-rollout baseline. |
| MVSR | `examples/3b_mvsr.sh` | Qwen2.5-VL-3B vanilla single-rollout baseline. |
| MVSR + text KL | `examples/7b_mvsr_text_kl.sh` | Adds text-only KL regularization. |
| MVSR + entropy loss | `examples/7b_mvsr_entropy_loss.sh` | Adds entropy loss regularization. |
<!-- | MVSR per-sample rho | `examples/7b_mvsr_per_sample_rho.sh` | Enables per-sample forgetting rates. | -->
| GRPO | `examples/grpo.sh`, `examples/3b_grpo.sh` | Group-rollout baselines. |
| REINFORCE++ | `examples/reinforce.sh`, `examples/3b_reinforce.sh` | REINFORCE-style baselines. |
| RLOO | `examples/rloo.sh`, `examples/3b_rloo.sh` | Leave-one-out baselines. |


## Installation

```bash
cd MSSR
pip install -e .
```


## Run MSSR

```bash
cd MSSR
bash examples/7b_mssr.sh
```

The main 7B MSSR script uses:

```bash
python -m verl.trainer.main \
  config=examples/config_mssr.yaml \
  data.train_files=Osilly/Vision-R1-rl@train \
  data.val_files=Osilly/Vision-R1-rl@test \
  algorithm.mvsr_run_initialization=true \
  algorithm.text_kl_enabled=false \
  algorithm.use_entropy_shaping=true \
  worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
  trainer.experiment_name=7b_mssr \
  trainer.n_gpus_per_node=8
```

For the 3B MSSR run:

```bash
bash examples/3b_mssr.sh
```

## Run MVSR Baselines

```bash
bash examples/7b_mvsr.sh
bash examples/3b_mvsr.sh
```

MVSR keeps the same single-rollout value-tracking machinery but does not enable the MSSR stabilization recipe by default.

## Configuration Reference

The main config is `examples/config_mssr.yaml`.

### MVSR Fields

| Field | Meaning |
| --- | --- |
| `algorithm.adv_estimator=mvsr` | Selects Multimodal Vanilla Single-Rollout. |
| `algorithm.mvsr_run_initialization` | Runs initial sampling to initialize prompt values. |
| `algorithm.mvsr_n_init` | Number of initialization responses per prompt. |
| `algorithm.mvsr_rho_min`, `algorithm.mvsr_rho_max` | Forgetting-rate bounds for the prompt value tracker. |
| `algorithm.mvsr_target_kl` | KL target for adaptive forgetting. |
| `algorithm.mvsr_v_init` | Fallback value for unseen prompts or skipped initialization. |
| `algorithm.mvsr_normalize_globally` | Normalizes advantages globally across the batch. |
<!-- | `algorithm.mvsr_per_sample_rho` | Enables per-sample forgetting rates. | -->
<!-- | `algorithm.mvsr_d_half` | Half-life scale for per-sample rho. | -->
<!-- | `algorithm.mvsr_use_uncertainty_weighting` | Enables uncertainty-weighted prioritized sampling. | -->
| `algorithm.mvsr_kl_window_size` | Window size for global KL tracking. |

### MSSR Stabilization Fields

| Field | Meaning |
| --- | --- |
| `algorithm.use_entropy_shaping` | Main MSSR scripts enable this. |
| `algorithm.entropy_alpha` | Weight for entropy-shaped advantages. |
| `algorithm.entropy_kappa` | Cap for the entropy shaping term relative to advantage magnitude. |
| `algorithm.use_entropy_loss` | Optional entropy loss regularizer for ablations. |
| `algorithm.entropy_coef` | Entropy loss coefficient. |
| `algorithm.text_kl_enabled` | Optional text-only KL regularization. |
<!-- | `worker.actor.max_grad_norm` | Stabilized MSSR scripts use tighter gradient clipping. | -->
| `worker.rollout.n=1` | Required for the single-rollout setting. |

## Data

Default config:

```yaml
data.train_files: Osilly/Vision-R1-rl@train
data.val_files: Osilly/Vision-R1-rl@test
data.prompt_key: problem
data.answer_key: answer
data.image_key: images
data.format_prompt: ./examples/format_prompt/math.jinja
```


## Evaluation And Checkpoint Merge

Run the local evaluation script after editing model/checkpoint paths:

```bash
bash examples/eval.sh
```

Merge checkpoints with:

```bash
python3 scripts/model_merger.py \
  --local_dir checkpoints/mssr/<experiment_name>/global_step_<step>/actor
```

<!-- ## Practical Notes

- Keep `worker.rollout.n=1` for MVSR and MSSR experiments.
- MVSR is the vanilla baseline; MSSR is the stabilized single-rollout method.
- `7b_visual_reweight_entropy_anneal.sh` contains experimental CLI fields that are not present in the current config.
- Vision-language models inherit [EasyR1](https://github.com/hiyouga/EasyR1)'s Ray, FSDP, vLLM, and Ulysses limitations. -->

This framework builds on [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl).

## Citation

If you use MSSR, please cite:

```bibtex
@article{liu2025stable,
  title={Stable and Efficient Single-Rollout RL for Multimodal Reasoning},
  author={Liu, Rui and Yu, Dian and Ke, Lei and Liu, Haolin and Zhou, Yujun and Liang, Zhenwen and Mi, Haitao and Tokekar, Pratap and Yu, Dong},
  journal={arXiv preprint arXiv:2512.18215},
  year={2025}
}
```
