# ft-agent

An agent project under active development.

## Setup

This project uses `uv` for Python environment and package management.

```powershell
uv sync
Copy-Item .env.example .env
```

Set `DEEPSEEK_API_KEY` in the local `.env` file. The `.env` file is ignored by Git.

## DeepSeek

The default DeepSeek settings are:

- `DEEPSEEK_BASE_URL=https://api.deepseek.com`
- `DEEPSEEK_MODEL=deepseek-v4-pro`

Run the API check:

```powershell
uv run ft-agent-check
```
