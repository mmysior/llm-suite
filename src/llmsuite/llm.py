from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Type

import instructor
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from .settings import get_settings
from .utils import format_anthropic_image_content, format_openai_image_content

load_dotenv()


class MessageBuilder(Protocol):
    def __call__(
        self, text: str, image_path: Optional[Path] = None, system_prompt: Optional[str] = None
    ) -> list[dict]: ...


class ChatFunc(Protocol):
    def __call__(self, messages: list[dict], **kwargs) -> str: ...


class ExtractFunc(Protocol):
    def __call__(self, messages: list[dict], schema: Type[BaseModel], **kwargs) -> Any: ...


type LLMClient = OpenAI | Anthropic
type CompletionFunc = Callable[[LLMClient, dict], str]


# ------------------------------------------------------------------------------
# Chatter function
# ------------------------------------------------------------------------------


def chatter(client: LLMClient) -> CompletionFunc:
    def get_openai_completion(client: OpenAI, completion_params: dict) -> str:
        try:
            completion = client.chat.completions.create(**completion_params)
            return completion.choices[0].message.content or ""
        except Exception as e:
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
            raise RuntimeError(f"Anthropic completion failed: {e}")

    if isinstance(client, OpenAI):
        return get_openai_completion
    elif isinstance(client, Anthropic):
        return get_anthropic_completion
    else:
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
        print(f"Initializing {provider} client")
        return initializer(settings)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_messages(
    text: str,
    provider: str,
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
        raise ValueError(f"Unsupported client for patching: {type(client)}")

    return patched_client.chat.completions.create(response_model=schema, **completion_params)


# ------------------------------------------------------------------------------
# LLMSuite Factory
# ------------------------------------------------------------------------------


def init_chat_model(model: Optional[str] = None, provider: Optional[str] = None):
    provider = provider or get_settings().default_provider
    model = model or get_settings().default_model

    class ChatModel:
        def __init__(self, provider: str, model: str):
            self._provider: str = provider
            self._model: str = model

            self.build_messages: MessageBuilder = partial(build_messages, provider=self._provider)
            self.chat: ChatFunc = partial(chat, provider=self._provider, model=self._model)
            self.extract: ExtractFunc = partial(extract, provider=self._provider, model=self._model)

    return ChatModel(provider, model)
