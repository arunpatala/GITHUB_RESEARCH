#!/bin/bash
# Launch vLLM model server in Docker
MODEL="${1:-unsloth/Qwen2.5-3B-Instruct}"
PORT="${2:-8000}"
MAX_MODEL_LEN="${3:-4096}"
GPUS="${4:-all}"

docker run -d --name vllm-eval \
  --gpus "$GPUS" \
  -p "$PORT":8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai \
  --model "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code

echo "vLLM serving $MODEL on port $PORT"
echo "Wait for model to load, then run: python infer.py --model $MODEL"
echo "To stop: docker rm -f vllm-eval"
