#!/bin/bash
# Launch vLLM model server in Docker
MODEL="${1:-unsloth/Qwen2.5-3B-Instruct}"
PORT="${2:-8000}"
MAX_MODEL_LEN="${3:-4096}"
GPUS="${4:-all}"

# Derive GPU device number for container name
if [[ "$GPUS" == *"device="* ]]; then
  DEV=$(echo "$GPUS" | grep -oP 'device=\K[0-9]+')
else
  DEV="all"
fi
NAME="vllm-gpu${DEV}"

set -x
exec docker run --rm --name "$NAME" \
  --gpus "$GPUS" \
  -p "$PORT":8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai \
  --model "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code
