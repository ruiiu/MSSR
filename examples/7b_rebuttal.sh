#!/bin/bash

# -------- CONFIGURATION --------
HEAD_NODE_IP="10.7.113.22"             # Head node IP
HEAD_NODE_PORT="6379"
WORKER_NODES=()  # Worker node IPs, "29.119.96.254" 29.232.224.137 "29.127.36.241" "29.191.211.78" 29.232.228.185
SSH_USER="root"
CONDA_ENV="easyr1"
NETWORK_INTERFACE="bond1"
RAY_GPU_COUNT=8

# -------- START RAY HEAD --------
# echo "[HEAD] Starting Ray head node..."

# export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
# export http_proxy="http://star-proxy.oa.com:3128"
# export https_proxy="http://star-proxy.oa.com:3128"

# export CUDA_VISIBLE_DEVICES=2,3 

export TMPDIR=/tmp/rui/mssr_tmp

# pkill -f python 
# ray stop > /dev/null 2>&1
# ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=6379 --num-gpus=$RAY_GPU_COUNT

# sleep 3

# -------- START RAY WORKERS (if any) --------
for NODE in "${WORKER_NODES[@]}"; do
  echo "[WORKER] Connecting to $NODE and starting Ray worker..."
  ssh -p 36000 ${SSH_USER}@$NODE "
    conda activate $CONDA_ENV
    export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
    export http_proxy="http://star-proxy.oa.com:3128"
    export https_proxy="http://star-proxy.oa.com:3128"
    ray start --address=${HEAD_NODE_IP}:${HEAD_NODE_PORT} --num-gpus=$RAY_GPU_COUNT
  "
done

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # Must be a multimodal model

# -------- EXPERIMENT 1: SPO with per-sample rho --------
echo "========================================="
echo "Starting Experiment 1: SPO per-sample rho"
echo "Time: $(date)"
echo "========================================="

python3 -m verl.trainer.main \
    config=examples/config_visual_spo.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    algorithm.spo_run_initialization=true \
    algorithm.text_kl_enabled=false \
    algorithm.spo_per_sample_rho=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=7b_spo_per_sample_rho \
    trainer.n_gpus_per_node=$RAY_GPU_COUNT

echo "Experiment 1 completed at $(date)"
echo ""

# -------- EXPERIMENT 2: SPO with fixed rho --------
echo "========================================="
echo "Starting Experiment 2: SPO fixed rho"
echo "Time: $(date)"
echo "========================================="

python3 -m verl.trainer.main \
    config=examples/config_visual_spo.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    algorithm.spo_run_initialization=true \
    algorithm.text_kl_enabled=false \
    algorithm.spo_use_fixed_rho=true \
    algorithm.use_entropy_shaping=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=7b_spo_fixed_rho \
    trainer.n_gpus_per_node=$RAY_GPU_COUNT

echo "Experiment 2 completed at $(date)"
echo ""

# -------- EXPERIMENT 3: MSSR beta dist value estimation --------
echo "========================================="
echo "Starting Experiment 3: MSSR beta dist value estimation "
echo "Time: $(date)"
echo "========================================="

python3 -m verl.trainer.main \
    config=examples/config_visual_spo.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    algorithm.spo_run_initialization=true \
    algorithm.text_kl_enabled=false \
    algorithm.use_entropy_shaping=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=7b_mssr \
    trainer.n_gpus_per_node=$RAY_GPU_COUNT

echo "Experiment 3 completed at $(date)"
echo ""

# -------- EXPERIMENT 4: MVSR with entropy loss --------
echo "========================================="
echo "Starting Experiment 4: MVSR entropy loss 0.05"
echo "Time: $(date)"
echo "========================================="

python3 -m verl.trainer.main \
    config=examples/config_visual_spo.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    algorithm.spo_run_initialization=true \
    algorithm.text_kl_enabled=false \
    algorithm.use_entropy_loss=true \
    algorithm.entropy_coef=0.05 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=7b_mvsr_entropy_loss_0.05 \
    trainer.n_gpus_per_node=$RAY_GPU_COUNT

echo "Experiment 4 completed at $(date)"
echo ""
echo "========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "Finished at: $(date)"
echo "========================================="
