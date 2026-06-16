from openai import APIError, AuthenticationError, BadRequestError

from ft_agent.llm import DeepSeekLLM


def main() -> None:
    llm = DeepSeekLLM()

    try:
        content = llm.chat(
            [
                {"role": "system", "content": "Reply with exactly: ok"},
                {"role": "user", "content": "Health check."},
            ],
            max_tokens=128,
            temperature=0,
        )
    except AuthenticationError as exc:
        raise SystemExit(
            "DeepSeek authentication failed. Check DEEPSEEK_API_KEY."
        ) from exc
    except BadRequestError as exc:
        raise SystemExit(f"DeepSeek request failed: {exc.message}") from exc
    except APIError as exc:
        raise SystemExit(f"DeepSeek API error: {exc.message}") from exc

    print(f"DeepSeek check passed. Model={llm.config.model}, response={content.strip()!r}")
