#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    algorithm.adv_estimator=rloo \
    trainer.experiment_name=7b_rloo_vision \
    trainer.n_gpus_per_node=8
