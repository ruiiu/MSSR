#!/bin/bash

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config_mssr.yaml \
    data.train_files=XenoZLH/MMRL30k@train \
    data.val_files=XenoZLH/MMRL30k@k12_test \
    algorithm.spo_run_initialization=true \
    algorithm.spo_enable_visual_reweighting=true \
    algorithm.use_entropy_shaping=true \
    algorithm.spo_visual_annealing=true \
    algorithm.spo_visual_annealing_start_prob=1.0 \
    algorithm.spo_visual_annealing_end_prob=0.0 \
    algorithm.spo_visual_annealing_start_step=0 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=7b_visual_reweight_entropy_anneal \
    trainer.n_gpus_per_node=8
