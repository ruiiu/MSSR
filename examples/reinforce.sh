#!/bin/bash

# -------- CONFIGURATION --------
HEAD_NODE_IP="29.232.228.185"             # Head node IP
HEAD_NODE_PORT="6379"
WORKER_NODES=()  # Worker node IPs, "29.119.96.254" 29.232.224.137 "29.127.36.241" "29.191.211.78" 29.232.228.185, 29.127.80.107, 29.160.40.86, 29.160.43.142
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
#     -- \
    python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    data.rollout_batch_size=1024 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=1 \
    worker.rollout.gpu_memory_utilization=0.5 \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.use_entropy_shaping=false \
    trainer.experiment_name=7b_reinforce_pp_vision \
    trainer.n_gpus_per_node=$RAY_GPU_COUNT \
    trainer.load_checkpoint_path=checkpoints/mm-spo/7b_reinforce_pp_vision/global_step_75

# To keep GPUs busy after training completes, run a placeholder process
nohup python ../matrix_multiplication_gpus.py --gpus 8 --size 5000 > /dev/null 2>&1 &