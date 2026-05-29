#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    data.rollout_batch_size=2048 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=1 \
    algorithm.adv_estimator=reinforce_plus_plus \
    trainer.experiment_name=3b_reinforce_pp \
    trainer.n_gpus_per_node=8
