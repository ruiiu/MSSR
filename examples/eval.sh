#!/bin/bash

# -------- CONFIGURATION --------
HEAD_NODE_IP="29.232.224.137"             # Head node IP
HEAD_NODE_PORT="6379"
WORKER_NODES=()  # Worker node IPs, "29.119.96.254" 29.232.224.137 "29.127.36.241" "29.191.211.78" 29.232.228.185
SSH_USER="root"
CONDA_ENV="easyr1"
NETWORK_INTERFACE="bond1"
RAY_GPU_COUNT=8

# -------- START RAY HEAD --------
echo "[HEAD] Starting Ray head node..."

export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

pkill -f python 
ray stop > /dev/null 2>&1
# ray start --head --dashboard-host=0.0.0.0 --num-gpus=$RAY_GPU_COUNT

sleep 3

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


# ray job submit \
#     --address=http://${HEAD_NODE_IP}:8265 \
#     --no-wait \
#     -- \
    python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=AI4Math/MathVerse:testmini@testmini \
    data.val_files=AI4Math/MathVerse:testmini@testmini \
    data.prompt_key=question \
    data.answer_key=answer \
    data.image_key=image \
    algorithm.spo_run_initialization=false \
    algorithm.text_kl_enabled=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=7b_mssr_mathverse_eval \
    trainer.load_checkpoint_path=checkpoints/mm-spo/7b_spo_entropy/global_step_120 \
    trainer.val_only=true \
    trainer.n_gpus_per_node=$RAY_GPU_COUNT



