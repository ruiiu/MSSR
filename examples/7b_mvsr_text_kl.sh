#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config_mssr.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    algorithm.mvsr_run_initialization=true \
    algorithm.kl_coef=0.00 \
    algorithm.text_kl_enabled=true \
    algorithm.text_kl_coef=0.01 \
    algorithm.use_entropy_loss=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=7b_mvsr_text_kl \
    trainer.n_gpus_per_node=8
