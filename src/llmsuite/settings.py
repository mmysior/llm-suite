from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PromptSettings(BaseSettings):
    templates_dir: str = Field(default="./prompts")


class LLMProviderSettings(BaseSettings):
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = 1024

    model_config = SettingsConfigDict(env_prefix="DEFAULT_")


class OpenAISettings(LLMProviderSettings):
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"

    model_config = SettingsConfigDict(env_prefix="OPENAI_")


class AnthropicSettings(LLMProviderSettings):
    api_key: str = ""

    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_")


class TogetherAISettings(LLMProviderSettings):
    api_key: str = ""
    base_url: str = "https://api.together.xyz/v1"

    model_config = SettingsConfigDict(env_prefix="TOGETHER_")


class PerplexitySettings(LLMProviderSettings):
    api_key: str = ""
    base_url: str = "https://api.perplexity.ai"

    model_config = SettingsConfigDict(env_prefix="PERPLEXITY_")


class GroqSettings(LLMProviderSettings):
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"

    model_config = SettingsConfigDict(env_prefix="GROQ_")


class OllamaSettings(LLMProviderSettings):
    api_key: str = "ollama"
    base_url: str = "http://localhost:11434/v1"

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")


class LMStudioSettings(LLMProviderSettings):
    api_key: str = "lmstudio"
    base_url: str = "http://localhost:1234/v1"

    model_config = SettingsConfigDict(env_prefix="LMSTUDIO_")


class Settings(BaseSettings):
    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    prompt: PromptSettings = PromptSettings()
    openai: OpenAISettings = OpenAISettings()
    ollama: OllamaSettings = OllamaSettings()
    groq: GroqSettings = GroqSettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    lmstudio: LMStudioSettings = LMStudioSettings()
    perplexity: PerplexitySettings = PerplexitySettings()
    together: TogetherAISettings = TogetherAISettings()


@lru_cache
def get_settings():
    return Settings()
