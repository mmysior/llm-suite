from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Centralize env file path
root_dir = Path(__file__).parent.parent.parent
env_file_path = root_dir / ".env"


class PromptSettings(BaseSettings):
    templates_dir: str = Field(default="./prompts")


class LLMProviderSettings(BaseSettings):
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    max_retries: int = 3

    model_config = SettingsConfigDict(
        env_file=env_file_path, env_file_encoding="utf-8", extra="ignore"
    )


class OpenAISettings(LLMProviderSettings):
    api_key: str | None = Field(alias="OPENAI_API_KEY", default=None)
    base_url: str = "https://api.openai.com/v1"


class AnthropicSettings(LLMProviderSettings):
    api_key: str | None = Field(alias="ANTHROPIC_API_KEY", default=None)


class TogetherAISettings(LLMProviderSettings):
    api_key: str | None = Field(alias="TOGETHER_API_KEY", default=None)
    base_url: str = "https://api.together.xyz/v1"


class PerplexitySettings(LLMProviderSettings):
    api_key: str | None = Field(alias="PERPLEXITY_API_KEY", default=None)
    base_url: str = "https://api.perplexity.ai"


class GroqSettings(LLMProviderSettings):
    api_key: str | None = Field(alias="GROQ_API_KEY", default=None)
    base_url: str = "https://api.groq.com/openai/v1"


class OllamaSettings(LLMProviderSettings):
    api_key: str = "ollama"
    base_url: str = "http://localhost:11434/v1"


class LMStudioSettings(LLMProviderSettings):
    api_key: str = "lmstudio"
    base_url: str = "http://localhost:1234/v1"


class Settings(BaseSettings):
    default_provider: str | None = Field(alias="DEFAULT_PROVIDER", default=None)
    default_model: Optional[str] = Field(alias="DEFAULT_MODEL", default=None)

    # Provider-specific settings
    prompt: PromptSettings = PromptSettings()
    openai: OpenAISettings = OpenAISettings()
    ollama: OllamaSettings = OllamaSettings()
    groq: GroqSettings = GroqSettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    lmstudio: LMStudioSettings = LMStudioSettings()
    perplexity: PerplexitySettings = PerplexitySettings()
    together: TogetherAISettings = TogetherAISettings()

    model_config = SettingsConfigDict(
        env_file=env_file_path, env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings():
    return Settings()
