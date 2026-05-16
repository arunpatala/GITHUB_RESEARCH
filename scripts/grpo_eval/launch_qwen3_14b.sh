#!/bin/bash
# Launch Qwen3-14B on 2 GPUs with tensor parallelism
set -x
exec docker run --rm --name vllm-qwen3-14b \
  --gpus '"device=2,3"' \
  -p 8002:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai \
  --model Qwen/Qwen3-14B \
  --max-model-len 8192 \
  --tensor-parallel-size 2 \
  --trust-remote-code
