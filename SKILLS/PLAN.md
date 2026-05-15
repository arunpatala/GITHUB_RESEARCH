# SKILLS Plan

## Reference Files

| File | Purpose |
|------|---------|
| 00_git.md | Git auth using PAT from `.secret`, remote config, commit/push workflow |

## Current Goal

Run unsloth fine-tuning notebooks as python scripts using the official Docker image.

### Steps

1. **Clone notebooks repo**
   ```bash
   git clone https://github.com/unslothai/notebooks.git
   ```
   - Key dir: `python_scripts/` (pre-converted .py versions of notebooks)
   - Also: `nb/` (original .ipynb notebooks)

2. **Pull & run unsloth Docker image**
   ```bash
   docker run -d \
     -v $(pwd):/workspace/work \
     --gpus all \
     unsloth/unsloth
   ```
   - Image: `unsloth/unsloth` (Docker Hub)
   - All deps pre-installed (torch, transformers, unsloth, trl, peft)
   - Runs as non-root `unsloth` user
   - Mount local dir to `/workspace/work`

3. **Pick a script and run inside container**
   ```bash
   docker exec -it <container_id> python /workspace/work/notebooks/python_scripts/<script>.py
   ```

4. **Candidate notebooks to try first**
   - Llama 3 fine-tuning (SFT)
   - GRPO / reinforcement learning
   - Vision fine-tuning

### Notes
- No Jupyter needed — run scripts directly
- GPU required (NVIDIA + nvidia-container-toolkit)
- Volume mount preserves outputs between runs
