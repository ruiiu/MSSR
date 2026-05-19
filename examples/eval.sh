#!/bin/bash

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=AI4Math/MathVerse:testmini@testmini \
    data.val_files=AI4Math/MathVerse:testmini@testmini \
    data.prompt_key=question \
    data.answer_key=answer \
    data.image_key=image \
    algorithm.spo_run_initialization=false \
    algorithm.text_kl_enabled=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=7b_mssr_mathverse_eval \
    trainer.load_checkpoint_path=checkpoints/mssr/7b_spo_entropy/global_step_120 \
    trainer.val_only=true \
    trainer.n_gpus_per_node=8
