#!/bin/bash

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main \
    config=examples/config_mssr.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    algorithm.spo_run_initialization=true \
    algorithm.text_kl_enabled=false \
    algorithm.use_entropy_shaping=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=3b_mssr \
    trainer.n_gpus_per_node=8 \
    trainer.load_checkpoint_path=checkpoints/mssr/3b_spo_entropy_vision/global_step_100

nohup python ../matrix_multiplication_gpus.py --gpus 8 --size 5000 > /dev/null 2>&1 &
