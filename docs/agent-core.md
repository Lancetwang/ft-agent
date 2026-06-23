# Agent Core Dependency

FT-Agent is built on top of the standalone `agent-core-runtime` package:

https://github.com/Lancetwang/agent-core-runtime

## Runtime Package

Reusable primitives come from the installed `agent_core` package:

- `agent_core.agent`: thin `Agent` runner
- `agent_core.core`: `Node`, `Flow`, `RunContext`, trace/runtime events, and flow results
- `agent_core.models`: provider-neutral `ChatModel` protocol
- `agent_core.nodes`: reusable model/tool loop nodes
- `agent_core.tools`: tool decorator, OpenAI-compatible tool schema, executor, file tools, and tool-call node

FT-Agent pins the dependency in `pyproject.toml` so the application and the upstream runtime stay reproducible.

## FT Application Layer

The Fischer-Tropsch catalyst application stays under `ft_agent`:

- `ft_agent.llm`
- `ft_agent.pipeline`
- `ft_agent.web`
- `ft_agent.util`

The `ft_agent.core`, `ft_agent.tools`, and `ft_agent.agent` modules are compatibility wrappers. They keep application imports stable while the runtime source of truth lives in `agent-core-runtime`.

## Merge Rule

Runtime changes should be made in the upstream `agent-core-runtime` repository. Do not modify runtime behavior inside FT-Agent.

FT-Agent may use `agent_core`, but should keep FT-specific behavior under `ft_agent`.

