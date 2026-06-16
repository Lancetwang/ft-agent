from dataclasses import dataclass
from os import getenv

from dotenv import load_dotenv


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"


def load_deepseek_config() -> DeepSeekConfig:
    load_dotenv()

    api_key = getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing. Create a local .env from .env.example.")

    return DeepSeekConfig(
        api_key=api_key,
        base_url=getenv("DEEPSEEK_BASE_URL", DeepSeekConfig.base_url),
        model=getenv("DEEPSEEK_MODEL", DeepSeekConfig.model),
    )
