# Unsloth Docker Fine-tuning

## Docker Image

```bash
docker run -d \
  -v $(pwd):/workspace/work \
  --gpus all \
  unsloth/unsloth
```

- Image: `unsloth/unsloth` (hub.docker.com/r/unsloth/unsloth)
- Pre-installed: torch, transformers, unsloth, trl, peft
- Runs as non-root `unsloth` user
- Container paths: `/workspace/work/` (mount), `/workspace/unsloth-notebooks/` (examples)

## Notebooks Repo

```bash
git clone https://github.com/unslothai/notebooks.git
```

Key directories:
- `python_scripts/` — pre-converted .py scripts (run directly)
- `nb/` — original .ipynb notebooks
- `kaggle/` — Kaggle-specific versions

## Running Scripts

```bash
docker exec -it <container_id> python /workspace/work/notebooks/python_scripts/<script>.py
```

## Requirements

- NVIDIA GPU
- nvidia-container-toolkit installed
- Docker Engine

## Env Vars

| Variable | Default | Purpose |
|----------|---------|---------|
| JUPYTER_PASSWORD | unsloth | Jupyter Lab password |
| JUPYTER_PORT | 8888 | Jupyter port |
| USER_PASSWORD | unsloth | sudo password |
