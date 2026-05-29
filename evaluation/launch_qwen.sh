#!/bin/bash

MODEL="Qwen/Qwen2.5-72B-Instruct"

python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --tensor-parallel-size 8 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code

