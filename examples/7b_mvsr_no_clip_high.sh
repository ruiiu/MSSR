#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config_mssr.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    algorithm.mvsr_run_initialization=true \
    algorithm.text_kl_enabled=false \
    worker.actor.clip_ratio_high=0.2 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=7b_mvsr_no_clip_high \
    trainer.n_gpus_per_node=8
