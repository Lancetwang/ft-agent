from openai import APIError, AuthenticationError, BadRequestError

from ft_agent.config import load_deepseek_config
from ft_agent.deepseek_client import create_deepseek_client


def main() -> None:
    config = load_deepseek_config()
    client = create_deepseek_client(config)

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "Reply with exactly: ok"},
                {"role": "user", "content": "Health check."},
            ],
            max_tokens=128,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except AuthenticationError as exc:
        raise SystemExit(
            "DeepSeek authentication failed. Check DEEPSEEK_API_KEY."
        ) from exc
    except BadRequestError as exc:
        raise SystemExit(f"DeepSeek request failed: {exc.message}") from exc
    except APIError as exc:
        raise SystemExit(f"DeepSeek API error: {exc.message}") from exc

    content = response.choices[0].message.content or ""
    print(f"DeepSeek check passed. Model={config.model}, response={content.strip()!r}")
