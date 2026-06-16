from openai import OpenAI

from ft_agent.config import DeepSeekConfig, load_deepseek_config


def create_deepseek_client(config: DeepSeekConfig | None = None) -> OpenAI:
    config = config or load_deepseek_config()
    return OpenAI(api_key=config.api_key, base_url=config.base_url)
