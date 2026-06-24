from __future__ import annotations

import sys

from ft_agent.pipeline import build_ft_agent


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Write an experiment report for cobalt FT catalyst stability."
    result = build_ft_agent().run({"question": question}, max_steps=24)
    print(result.payload.get("answer", ""))
    print()
    print("path:", " -> ".join(result.path))


if __name__ == "__main__":
    main()
