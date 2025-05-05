import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class LLMProviderSettings(BaseSettings):
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = 1024
    max_retries: int = 3
    default_model: str = ""


class OpenAISettings(LLMProviderSettings):
    default_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    vector_size: int = 1536
    base_url: str = "https://api.openai.com/v1"
    api_key: str = os.getenv("OPENAI_API_KEY", "")


class OllamaSettings(LLMProviderSettings):
    default_model: str = "llama3.1:8b"
    embedding_model: str = "mxbai-embed-large:335m"
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
    api_key: str = "ollama"


class LMStudioSettings(LLMProviderSettings):
    default_model: str = "gemma-2-2b-it"
    base_url: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    api_key: str = "lmstudio"


class PerplexitySettings(LLMProviderSettings):
    default_model: str = "sonar"
    base_url: str = "https://api.perplexity.ai"
    api_key: str = os.getenv("PERPLEXITY_API_KEY", "")
    temperature: float = 0.2


class GroqSettings(LLMProviderSettings):
    default_model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = os.getenv("GROQ_API_KEY", "")


class AnthropicSettings(LLMProviderSettings):
    default_model: str = "claude-3-7-sonnet-20250219"
    base_url: str = "https://api.anthropic.com/v1"
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")


class Settings(BaseSettings):
    openai: OpenAISettings = OpenAISettings()
    ollama: OllamaSettings = OllamaSettings()
    groq: GroqSettings = GroqSettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    lmstudio: LMStudioSettings = LMStudioSettings()
    perplexity: PerplexitySettings = PerplexitySettings()


@lru_cache
def get_settings():
    return Settings()
