import logging
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Type

import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from .settings import get_settings
from .utils import format_anthropic_image_content, format_openai_image_content

logger = logging.getLogger(__name__)


type LLMClient = OpenAI | Anthropic
type CompletionFunc = Callable[[LLMClient, dict], str]


class ChatModelProtocol(Protocol):
    def build_messages(
        self, text: str, image_path: Optional[Path] = None, system_prompt: Optional[str] = None
    ) -> list[dict]: ...

    def chat(self, messages: list[dict], **kwargs) -> str: ...

    def extract(self, messages: list[dict], schema: Type[BaseModel], **kwargs) -> Any: ...


# ------------------------------------------------------------------------------
# Chatter function
# ------------------------------------------------------------------------------


def chatter(client: LLMClient) -> CompletionFunc:
    def get_openai_completion(client: OpenAI, completion_params: dict) -> str:
        try:
            completion = client.chat.completions.create(**completion_params)
            return completion.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise RuntimeError(f"OpenAI completion failed: {e}")

    def get_anthropic_completion(client: Anthropic, completion_params: dict) -> str:
        try:
            params = completion_params.copy()
            messages = params.pop("messages")

            if messages and messages[0]["role"] == "system":
                params["system"] = messages[0]["content"]
                messages = messages[1:]

            completion = client.messages.create(messages=messages, **params)
            return completion.content[0].text
        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            raise RuntimeError(f"Anthropic completion failed: {e}")

    if isinstance(client, OpenAI):
        return get_openai_completion
    elif isinstance(client, Anthropic):
        return get_anthropic_completion
    else:
        logger.error(f"Unsupported client type: {type(client)}")
        raise ValueError(f"Unsupported client type: {type(client)}")


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def get_client(provider: str) -> LLMClient:
    settings = getattr(get_settings(), provider)

    client_initializers = {
        "openai": lambda s: OpenAI(api_key=s.api_key),
        "ollama": lambda s: OpenAI(base_url=s.base_url, api_key=s.api_key),
        "groq": lambda s: OpenAI(base_url=s.base_url, api_key=s.api_key),
        "perplexity": lambda s: OpenAI(base_url=s.base_url, api_key=s.api_key),
        "lmstudio": lambda s: OpenAI(base_url=s.base_url, api_key=s.api_key),
        "anthropic": lambda s: Anthropic(api_key=s.api_key),
        "together": lambda s: OpenAI(base_url=s.base_url, api_key=s.api_key),
    }

    initializer = client_initializers.get(provider)
    if initializer:
        logger.debug(f"Initializing {provider} client")
        return initializer(settings)
    logger.error(f"Unsupported LLM provider: {provider}")
    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_messages(
    provider: str,
    text: str,
    image_path: Optional[Path] = None,
    system_prompt: Optional[str] = None,
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    if not image_path:
        messages.append({"role": "user", "content": text})
    else:
        if provider == "anthropic":
            messages.extend(format_anthropic_image_content(text, image_path))
        else:
            messages.extend(format_openai_image_content(text, image_path))
    return messages


# ------------------------------------------------------------------------------
# Completion functions
# ------------------------------------------------------------------------------


def chat(messages: list[dict], model: str, provider: str, **kwargs) -> str:
    logger.debug(f"Starting chat with model: {model}, provider: {provider}")
    settings = getattr(get_settings(), provider)
    client = get_client(provider)

    completion_params = {
        "model": model,
        "temperature": kwargs.get("temperature", settings.temperature),
        "top_p": kwargs.get("top_p", settings.top_p),
        "max_tokens": kwargs.get("max_tokens", settings.max_tokens),
        "messages": messages,
    }

    completion_func = chatter(client)
    return completion_func(client, completion_params)


def extract(
    messages: list[dict], schema: Type[BaseModel], model: str, provider: str, **kwargs
) -> Any:
    logger.debug(
        f"Starting extraction with model: {model}, provider: {provider}, schema: {schema.__name__}"
    )
    settings = getattr(get_settings(), provider)
    client = get_client(provider)

    completion_params = {
        "model": model,
        "temperature": kwargs.get("temperature", settings.temperature),
        "top_p": kwargs.get("top_p", settings.top_p),
        "max_tokens": kwargs.get("max_tokens", settings.max_tokens),
        "messages": messages,
    }

    if isinstance(client, OpenAI):
        patched_client = instructor.from_openai(client)
    elif isinstance(client, Anthropic):
        patched_client = instructor.from_anthropic(client)
    else:
        logger.error(f"Unsupported client for patching: {type(client)}")
        raise ValueError(f"Unsupported client for patching: {type(client)}")

    return patched_client.chat.completions.create(response_model=schema, **completion_params)


# ------------------------------------------------------------------------------
# LLMSuite Factory
# ------------------------------------------------------------------------------


def init_chat_model(
    model: Optional[str] = None, provider: Optional[str] = None
) -> ChatModelProtocol:
    provider = provider or get_settings().default_provider
    if not provider:
        raise ValueError("Provider must be specified or set in DEFAULT_PROVIDER env variable.")
    model = model or get_settings().default_model
    if not model:
        raise ValueError("Model must be specified or set in DEFAULT_MODEL env variable.")

    logger.debug(f"Initializing chat model with provider: {provider}, model: {model}")

    class ChatModel:
        def __init__(self, provider: str, model: str):
            self._provider = provider
            self._model = model

        def build_messages(
            self, text: str, image_path: Optional[Path] = None, system_prompt: Optional[str] = None
        ) -> list[dict]:
            return build_messages(self._provider, text, image_path, system_prompt)

        def chat(self, messages: list[dict], **kwargs) -> str:
            return chat(messages, self._model, self._provider, **kwargs)

        def extract(self, messages: list[dict], schema: Type[BaseModel], **kwargs) -> Any:
            return extract(messages, schema, self._model, self._provider, **kwargs)

    return ChatModel(provider, model)
