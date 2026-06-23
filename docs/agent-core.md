# agent-core Branch

This branch separates the reusable agent runtime from the FT-Agent application.

## Runtime Package

Reusable primitives live under `agent_core`:

- `agent_core.agent`: thin `Agent` runner
- `agent_core.core`: `Node`, `Flow`, trace events, and flow results
- `agent_core.tools`: tool decorator, OpenAI-compatible tool schema, executor, file tools, and tool-call node

Application code should depend on `agent_core` directly when it needs runtime primitives.

## FT Application Layer

The Fischer-Tropsch catalyst application stays under `ft_agent`:

- `ft_agent.llm`
- `ft_agent.pipeline`
- `ft_agent.web`
- `ft_agent.util`

The `ft_agent.core`, `ft_agent.tools`, and `ft_agent.agent` modules are compatibility wrappers. They keep old imports working while the runtime source of truth moves to `agent_core`.

## Merge Rule

Changes to `agent_core` should remain application-agnostic. Do not import `ft_agent`, DeepSeek config, FT pipeline nodes, or web UI code from the runtime package.

Changes to `ft_agent` may use `agent_core`, but should not modify runtime behavior unless the change also belongs in `agent_core`.

