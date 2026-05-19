#!/bin/bash

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Osilly/Vision-R1-rl@train \
    data.val_files=Osilly/Vision-R1-rl@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    algorithm.adv_estimator=grpo \
    trainer.experiment_name=7b_grpo_vision_150 \
    trainer.n_gpus_per_node=8 \
    trainer.load_checkpoint_path=checkpoints/mssr/7b_grpo_vision/global_step_120

nohup python ../matrix_multiplication_gpus.py --gpus 8 --size 5000 > /dev/null 2>&1 &
