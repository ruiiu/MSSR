#!/bin/bash

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

MODEL="Qwen/Qwen2.5-72B-Instruct"
# MODEL="Qwen/Qwen3-32B"

python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --tensor-parallel-size 8 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code

python ../matrix_multiplication_gpus.py --gpus 8 --size 5000
