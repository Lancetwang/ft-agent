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
- `DEEPSEEK_MODEL=deepseek-v4-flash`

Run the API check:

```powershell
uv run ft-agent-check
```

## Project Shape

The project is organized around a small node-flow agent runtime:

- `src/ft_agent/core/`: node and flow abstractions
- `src/ft_agent/llm/`: model calls and LLM-backed nodes
- `src/ft_agent/agent.py`: thin agent runner over a flow
- `examples/`: runnable sketches for flow composition

Each node returns a `NodeResult` with one `route`. The `Flow` resolves that route through its transition table, so each case maps to at most one next node.

Run the local flow example:

```powershell
uv run python examples/basic_flow.py
```
