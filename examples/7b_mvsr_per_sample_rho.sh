#!/bin/bash

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config_mssr.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    algorithm.mvsr_run_initialization=true \
    algorithm.text_kl_enabled=false \
    algorithm.mvsr_per_sample_rho=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=7b_mvsr_per_sample_rho \
    trainer.n_gpus_per_node=8
