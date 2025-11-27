# EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)](https://github.com/hiyouga/EasyR1/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)

### Used by [Amazon Web Services](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/)

This project is a clean fork of the original [veRL](https://github.com/volcengine/verl) project to support vision language models, we thank all the authors for providing such a high-performance RL training framework.

EasyR1 is efficient and scalable due to the design of **[HybirdEngine](https://arxiv.org/abs/2409.19256)** and the latest release of **[vLLM](https://github.com/vllm-project/vllm)**'s SPMD mode.

## Features

- Supported models
  - Llama3/Qwen2/Qwen2.5/Qwen3 language models
  - Qwen2/Qwen2.5-VL vision language models
  - DeepSeek-R1 distill models

- Supported algorithms
  - GRPO
  - DAPO
  - Reinforce++
  - ReMax
  - RLOO
  - **SPO (Single-stream Policy Optimization)** - Novel algorithm with persistent value tracking and multimodal support

- Supported datasets
  - Any text, vision-text dataset in a [specific format](#custom-dataset)

- Supported tricks
  - Padding-free training
  - Resuming from checkpoint
  - Wandb & SwanLab & Mlflow & Tensorboard tracking

## Requirements

### Software Requirements

- Python 3.9+
- transformers>=4.51.0
- flash-attn>=2.4.3
- vllm>=0.8.3

We provide a [Dockerfile](./Dockerfile) to easily build environments.

We recommend using the [pre-built docker image](https://hub.docker.com/r/hiyouga/verl) in EasyR1.

```bash
docker pull hiyouga/verl:ngc-th2.7.0-cu12.6-vllm0.9.1
```

### Hardware Requirements

\* *estimated*

| Method                   | Bits |  1.5B  |   3B   |   7B   |   32B   |   72B   |
| ------------------------ | ---- | ------ | ------ | ------ | ------- | ------- |
| GRPO Full Fine-Tuning    |  AMP | 2*24GB | 4*40GB | 8*40GB | 16*80GB | 32*80GB |
| GRPO Full Fine-Tuning    | BF16 | 1*24GB | 1*40GB | 4*40GB |  8*80GB | 16*80GB |

> [!NOTE]
> Use `worker.actor.fsdp.torch_dtype=bf16` and `worker.actor.optim.strategy=adamw_bf16` to enable bf16 training.
>
> We are working hard to reduce the VRAM in RL training, LoRA support will be integrated in next updates.

## Tutorial: Run Qwen2.5-VL GRPO on [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k) Dataset in Just 3 Steps

![image](assets/qwen2_5_vl_7b_geo.png)

### Installation

```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
```

### GRPO Training

```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

> [!TIP]
> If you encounter issues with connecting to Hugging Face, consider using `export HF_ENDPOINT=https://hf-mirror.com`.
>
> If you want to use SwanLab logger, consider using `bash examples/qwen2_5_vl_7b_geo3k_swanlab.sh`.

## SPO (Single-stream Policy Optimization): Accurate Implementation

We have implemented the **Single-stream Policy Optimization (SPO)** algorithm based on the paper [arXiv:2509.13232](https://arxiv.org/abs/2509.13232) with **complete accuracy** to the paper's specifications. SPO addresses key limitations of group-based methods like GRPO by:

- **Eliminating degenerate groups** through per-prompt persistent value tracking
- **Removing synchronization barriers** via single-stream operation (n=1) for 4.35× throughput speedup
- **Providing stable learning signals** with per-prompt baselines and global batch normalization
- **Supporting both text-only and multimodal scenarios** with unified Bayesian value tracking

### SPO Training Examples

#### Text-only SPO on Math Reasoning
```bash
bash examples/qwen2_5_7b_math_spo.sh
```

#### Vision-Language SPO on Geometry Problems
```bash
bash examples/qwen2_5_vl_7b_geo3k_spo.sh
```

### Key SPO Features (Section 4.1 - Paper-Accurate)

1. **Beta Distribution Value Function**: V(x) ~ Beta(α(x), β(x)) with V̂(x) = α/(α+β) (Equation 5)
2. **Per-Prompt Informed Initialization (Algorithm 2)**: 
   - For each prompt x, collect n_0=8 outcomes with initial policy π_0
   - Compute v̂_0(x) = (1/n_0) * Σ r^(k) (average of outcomes)
   - Set α_0(x) = N_0 * v̂_0(x), β_0(x) = N_0 * (1 - v̂_0(x)) where N_0 = 1/(1-ρ_min) = 8
3. **Bayesian Updates with Forgetting**: α_new = ρ*α + r, β_new = ρ*β + (1-r) (Equation 7)
4. **KL-Adaptive Forgetting Rates**: ρ ∈ [0.875, 0.96] adapts based on policy change rate
5. **Global Batch Normalization**: Normalizes advantages across entire batch after per-prompt baseline subtraction
6. **Single-Stream Operation**: Uses n=1 (one response per prompt) unlike GRPO's n>1 groups
7. **Unified Multimodal Support**: Same Beta-Bernoulli formulation works for text-only and vision-language

**Algorithm Initialization**: The trainer implements Algorithm 2 from Appendix A when `spo_run_initialization=true` (default). At the start of training, it collects `n_0` samples per prompt using the initial policy, computes per-prompt success rates v̂_0(x), and initializes the value tracker with α_0(x) = N_0 * v̂_0(x) and β_0(x) = N_0 * (1 - v̂_0(x)). This provides accurate, data-driven initialization. Set `spo_run_initialization=false` to skip initialization and use fallback (v_init=0.5 for all prompts) for faster startup, useful for debugging.

### Implementation Highlights

From the paper (Section D: Training and Evaluation Details):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `run_init` | true | Run Algorithm 2 initialization (true=paper-accurate, false=faster) |
| `N_0` | 8 | Effective window size: 1/(1-ρ_min) (Algorithm 2) |
| `n_0` | 2-5 | Samples per prompt for initialization (if run_init=true) |
| `v̂_0(x)` | computed | Per-prompt: (1/n_0) * Σ r^(k) (if run_init=true) |
| `α_0(x)` | N_0 * v̂_0(x) | Initial α for prompt x (if run_init=true) |
| `β_0(x)` | N_0 * (1-v̂_0(x)) | Initial β for prompt x (if run_init=true) |
| `v_fallback` | 0.5 | Fallback (if run_init=false or for unseen prompts) |
| `ρ_min` | 0.875 | Minimum forgetting rate (W=8, fast) |
| `ρ_max` | 0.96 | Maximum forgetting rate (W=25, slow) |
| `n` | 1 | Single response per prompt |
| `batch_size` | 2048 | 8× GRPO prompt batch (matches total compute) |
| `temperature` | 1.0 | Training sampling temperature |
| `top_p` | 1.0 | Training top-p sampling |

### Performance Improvements (from paper)

On hard math benchmarks with Qwen3-8B:
- **Average maj@32**: +3.4 pp over GRPO
- **BRUMO 25**: +7.3 pp absolute gain
- **AIME 25**: +4.4 pp absolute gain
- **HMMT 25**: +3.3 pp absolute gain
- **Consistent pass@k gains** across all evaluated k values

### Documentation

See [SPO_IMPLEMENTATION.md](SPO_IMPLEMENTATION.md) for detailed implementation documentation, including:
- Per-prompt value tracking architecture
- Bayesian update mechanism with adaptive forgetting
- Global batch normalization algorithm
- Comparison with GRPO
- Complete parameter specifications

The SPO implementation uses `config_spo.yaml` with paper-accurate hyperparameters that work for both text-only and vision-language scenarios.

## Custom Dataset

Please refer to the example datasets to prepare your own dataset.

- Text dataset: https://huggingface.co/datasets/hiyouga/math12k
- Image-text dataset: https://huggingface.co/datasets/hiyouga/geometry3k
- Multi-image-text dataset: https://huggingface.co/datasets/hiyouga/journeybench-multi-image-vqa
- Text-image mixed dataset: https://huggingface.co/datasets/hiyouga/rl-mixed-dataset

## How to Understand GRPO in EasyR1

![image](assets/easyr1_grpo.png)

- To learn about the GRPO algorithm, you can refer to [Hugging Face's blog](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer).

## How to Run 70B+ Model in Multi-node Environment

1. Start the Ray head node.

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

2. Start the Ray worker node and connect to the head node.

```bash
ray start --address=<head_node_ip>:6379
```

3. Check the Ray resource pool.

```bash
ray status
```

4. Run training script on the Ray head node only.

```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

See the **[veRL's official doc](https://verl.readthedocs.io/en/latest/start/multinode.html)** for more details about multi-node training and Ray debugger.

## Other Baselines

We also reproduced the following two baselines of the [R1-V](https://github.com/deep-agent/R1-V) project.
- [CLEVR-70k-Counting](examples/baselines/qwen2_5_vl_3b_clevr.sh): Train the Qwen2.5-VL-3B-Instruct model on counting problem.
- [GeoQA-8k](examples/baselines/qwen2_5_vl_3b_geoqa8k.sh): Train the Qwen2.5-VL-3B-Instruct model on GeoQA problem.

## Performance Baselines

See [baselines.md](assets/baselines.md).

## Awesome Work using EasyR1

- **MMR1**: Advancing the Frontiers of Multimodal Reasoning. [![[code]](https://img.shields.io/github/stars/LengSicong/MMR1)](https://github.com/LengSicong/MMR1)
- **Vision-R1**: Incentivizing Reasoning Capability in Multimodal Large Language Models. [![[code]](https://img.shields.io/github/stars/Osilly/Vision-R1)](https://github.com/Osilly/Vision-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.06749-blue)](https://arxiv.org/abs/2503.06749)
- **Seg-Zero**: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement. [![[code]](https://img.shields.io/github/stars/dvlab-research/Seg-Zero)](https://github.com/dvlab-research/Seg-Zero) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.06520-blue)](https://arxiv.org/abs/2503.06520)
- **MetaSpatial**: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse. [![[code]](https://img.shields.io/github/stars/PzySeere/MetaSpatial)](https://github.com/PzySeere/MetaSpatial) [![[arxiv]](https://img.shields.io/badge/arxiv-2503.18470-blue)](https://arxiv.org/abs/2503.18470)
- **Temporal-R1**: Envolving Temporal Reasoning Capability into LMMs via Temporal Consistent Reward. [![[code]](https://img.shields.io/github/stars/appletea233/Temporal-R1)](https://github.com/appletea233/Temporal-R1)
- **NoisyRollout**: Reinforcing Visual Reasoning with Data Augmentation. [![[code]](https://img.shields.io/github/stars/John-AI-Lab/NoisyRollout)](https://github.com/John-AI-Lab/NoisyRollout) [![[arxiv]](https://img.shields.io/badge/arxiv-2504.13055-blue)](https://arxiv.org/pdf/2504.13055)
- **GUI-R1**: A Generalist R1-Style Vision-Language Action Model For GUI Agents. [![[code]](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)](https://github.com/ritzz-ai/GUI-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2504.10458-blue)](https://arxiv.org/abs/2504.10458)
- **R1-Track**: Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/Wangbiao2/R1-Track)](https://github.com/Wangbiao2/R1-Track)
- **VisionReasoner**: Unified Visual Perception and Reasoning via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/dvlab-research/VisionReasoner)](https://github.com/dvlab-research/VisionReasoner) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.12081-blue)](https://arxiv.org/abs/2505.12081)
- **MM-UPT**: Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO. [![[code]](https://img.shields.io/github/stars/waltonfuture/MM-UPT)](https://github.com/waltonfuture/MM-UPT) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22453-blue)](https://arxiv.org/pdf/2505.22453)
- **RL-with-Cold-Start**: Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start. [![[code]](https://img.shields.io/github/stars/waltonfuture/RL-with-Cold-Start)](https://github.com/waltonfuture/RL-with-Cold-Start) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22334-blue)](https://arxiv.org/pdf/2505.22334)
- **ViGoRL**: Grounded Reinforcement Learning for Visual Reasoning. [![[code]](https://img.shields.io/github/stars/Gabesarch/grounded-rl)](https://github.com/Gabesarch/grounded-rl) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.22334-blue)](https://arxiv.org/abs/2505.23678)
- **Revisual-R1**: Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/CSfufu/Revisual-R1)](https://github.com/CSfufu/Revisual-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2506.04207-blue)](https://arxiv.org/abs/2506.04207)
- **SophiaVL-R1**: Reinforcing MLLMs Reasoning with Thinking Reward. [![[code]](https://img.shields.io/github/stars/kxfan2002/SophiaVL-R1)](https://github.com/kxfan2002/SophiaVL-R1) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.17018-blue)](https://arxiv.org/abs/2505.17018)
- **Vision-Matters**: Simple Visual Perturbations Can Boost Multimodal Math Reasoning. [![[code]](https://img.shields.io/github/stars/YutingLi0606/Vision-Matters)](https://github.com/YutingLi0606/Vision-Matters) [![[arxiv]](https://img.shields.io/badge/arxiv-2506.09736-blue)](https://arxiv.org/abs/2506.09736)
- **VTool-R1**: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use. [![[code]](https://img.shields.io/github/stars/VTOOL-R1/vtool-r1)](https://github.com/VTOOL-R1/vtool-r1) [![[arxiv]](https://img.shields.io/badge/arxiv-2505.19255-blue)](https://arxiv.org/abs/2505.19255)

## TODO

- Support LoRA (high priority).
- Support ulysses parallelism for VLMs (middle priority).
- Support more VLM architectures.

> [!NOTE]
> We will not provide scripts for supervised fine-tuning and inference in this project. If you have such requirements, we recommend using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Known bugs

These features are temporarily disabled for now, we plan to fix them one-by-one in the future updates.

- Vision language models are not compatible with ulysses parallelism yet.

## Discussion Group

👋 Join our [WeChat group](assets/wechat.jpg).

## FAQs

> ValueError: Image features and image tokens do not match: tokens: 8192, features 9800

Increase the `data.max_prompt_length` or reduce the `data.max_pixels`.

> RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:62

Reduce the `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`.

> RuntimeError: 0 active drivers ([]). There should only be one.

Uninstall `deepspeed` from the current python environment.

## Citation

Core contributors: [Yaowei Zheng](https://github.com/hiyouga), [Junting Lu](https://github.com/AL-377), [Shenzhi Wang](https://github.com/Shenzhi-Wang), [Zhangchi Feng](https://github.com/BUAADreamer), [Dongdong Kuang](https://github.com/Kuangdd01) and Yuwen Xiong

We also thank Guangming Sheng and Chi Zhang for helpful discussions.

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

For the SPO algorithm implementation:

```bibtex
@article{xu2025single,
  title={Single-stream Policy Optimization},
  author={Xu, Zhongwen and Ding, Zihan},
  journal={arXiv preprint arXiv:2509.13232},
  year={2025}
}
```

We recommend to also cite the original work.

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```
