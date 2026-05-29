#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

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
    trainer.n_gpus_per_node=8
