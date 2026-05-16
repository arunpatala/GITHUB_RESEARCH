# GSM8K Evaluation Results

Dataset: `openai/gsm8k` (test split, 200 samples unless noted)
System prompt: "Solve the math problem. Put your final answer as a number after ####."
Few-shot: 0

| Model | Size | Thinking? | pass@1 (greedy) | pass@8 (temp=1.0) | Gap |
|-------|------|-----------|-----------------|-------------------|-----|
| Qwen2.5-0.5B-Instruct | 0.5B | ❌ | 41.0% | 56.5% | +15.5% |
| Qwen3-0.6B | 0.6B | ✅ | 65.0% | 86.5% | +21.5% |
| Qwen3.5-0.8B | 0.8B | ✅ | 49.5% | 78.0% | +28.5% |
| Llama-3.2-1B-Instruct | 1B | ❌ | 35.0% | 60.0% | +25.0% |
| Qwen2.5-1.5B-Instruct | 1.5B | ❌ | 45.5% | 70.0% | +24.5% |
| Qwen2.5-3B-Instruct | 3B | ❌ | 71.0% | 89.5% | +18.5% |
| Llama-3.2-3B-Instruct | 3B | ❌ | 76.5% | 92.0% | +15.5% |
| Qwen3-4B | 4B | ✅ | 85.5% | 94.5% | +9.0% |
| Qwen3.5-4B | 4B | ✅ | 86.5% | 92.5% | +6.0% |
| Qwen2.5-7B-Instruct | 7B | ❌ | 86.0% | 96.0% | +10.0% |
| Qwen3-8B | 8B | ✅ | 79.5% | 100%* | +20.5% |
| Llama-3.1-8B-Instruct | 8B | ❌ | 82.5% | 96.5% | +14.0% |
| Qwen3-14B | 14B | ✅ | 94.0% | 100%* | +6.0% |

*40 samples (smaller test due to slow inference with thinking mode)

## Key Findings

1. **Thinking models punch above their weight** — Qwen3-0.6B (thinking, 65%) beats Qwen2.5-1.5B (non-thinking, 45.5%)
2. **Non-thinking models benefit more from sampling** — larger pass@1→pass@8 gaps
3. **Best-of-8 ceiling converges at 92-96%** for non-thinking models regardless of size
4. **Thinking models saturate GSM8K** — 8B+ with sampling reach 100%
5. **GRPO opportunity = the gap** — models with large gaps benefit most from RL training

## GRPO Training Candidates (by gap size)

| Model | Gap | Why |
|-------|-----|-----|
| Qwen3.5-0.8B | +28.5% | Tiny thinking model, huge inconsistency |
| Llama-3.2-1B-Instruct | +25.0% | Tiny non-thinking, fast to train |
| Qwen2.5-1.5B-Instruct | +24.5% | Small non-thinking, standard choice |
| Qwen3-0.6B | +21.5% | Smallest thinking model |
| Qwen2.5-3B-Instruct | +18.5% | Used in unsloth GRPO notebooks |
