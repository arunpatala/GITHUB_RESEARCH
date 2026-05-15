# Git Setup

## Authentication

PAT (Personal Access Token) is stored in `.secret` (gitignored).

## Configure remote with PAT

```bash
PAT=$(cat .secret)
git remote set-url origin https://arunpatala:${PAT}@github.com/arunpatala/GITHUB_RESEARCH.git
```

## Commit and push

```bash
git add <files>
git commit -m "message"
git push
```
