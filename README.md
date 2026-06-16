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

- `src/ft_agent/core/node.py`: node and flow abstractions
- `src/ft_agent/llm/`: model calls and LLM-backed nodes
- `src/ft_agent/tools/`: tool definitions, tool execution, and tool-call nodes
- `src/ft_agent/agent.py`: thin agent runner over a flow
- `src/ft_agent/util/`: project utilities and checks
- `examples/`: runnable sketches for flow composition

Each node returns `(action, payload)`. The `Flow` follows `node.successors[action]`, so each action maps to at most one next node.

```python
classify_node - "question" >> answer_question_node
classify_node - "statement" >> answer_statement_node
```

Run the local flow example:

```powershell
uv run python examples/basic_flow.py
```

Run the chatbot example:

```powershell
uv run python examples/chatbot.py
uv run python examples/chatbot.py "hello"
```

Run the mocked weather tool flow:

```powershell
uv run python examples/weather_tool_flow.py
```

Run the multi-turn tool chatbot:

```powershell
uv run python examples/tool_chatbot.py
```
