from pathlib import Path
from typing import Any, Callable, Optional, Type, cast

import instructor
from anthropic import Anthropic
from dotenv import load_dotenv
from instructor import Instructor
from openai import OpenAI
from pydantic import BaseModel

from llmsuite.config.settings import LLMProviderSettings, get_settings
from llmsuite.utils import format_anthropic_image_content, format_openai_image_content

load_dotenv()

type LLMClient = OpenAI | Anthropic
type CompletionFunc = Callable[[LLMClient, dict], str]


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def chatter(client: LLMClient) -> CompletionFunc:
    def get_openai_completion(client: OpenAI, completion_params: dict) -> str:
        completion = client.chat.completions.create(**completion_params)
        return completion.choices[0].message.content

    def get_anthropic_completion(client: Anthropic, completion_params: dict) -> str:
        messages = completion_params.pop("messages")
        if messages and messages[0]["role"] == "system":
            completion_params["system"] = messages[0]["content"]
            messages = messages[1:]
        completion = client.messages.create(messages=messages, **completion_params)
        return completion.content[0].text

    if isinstance(client, OpenAI):
        return cast(CompletionFunc, get_openai_completion)
    return cast(CompletionFunc, get_anthropic_completion)


# ------------------------------------------------------------------------------
# LLMService
# ------------------------------------------------------------------------------


class LLMService:
    def __init__(self, provider: str):
        self.provider: str = provider
        self.settings: LLMProviderSettings = getattr(get_settings(), provider)
        self.default_model: str = self.settings.default_model
        self.client: LLMClient = self._initialize_client()

    def _initialize_client(self) -> LLMClient:
        client_initializers = {
            "openai": lambda s: OpenAI(api_key=s.api_key),
            "ollama": lambda s: OpenAI(base_url=s.base_url, api_key=s.api_key),
            "groq": lambda s: OpenAI(base_url=s.base_url, api_key=s.api_key),
            "perplexity": lambda s: OpenAI(base_url=s.base_url, api_key=s.api_key),
            "lmstudio": lambda s: OpenAI(base_url=s.base_url, api_key=s.api_key),
            "anthropic": lambda s: Anthropic(api_key=s.api_key),
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _get_patched_client(self) -> Instructor:
        if isinstance(self.client, OpenAI):
            return instructor.from_openai(self.client)
        elif isinstance(self.client, Anthropic):
            return instructor.from_anthropic(self.client)
        else:
            raise ValueError(f"Unsupported client for patching: {type(self.client)}")

    def build_messages(
        self,
        text: str,
        image_path: Optional[Path] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        if not image_path:
            messages.append({"role": "user", "content": text})
        else:
            if self.provider == "anthropic":
                messages.extend(format_anthropic_image_content(text, image_path))
            else:
                messages.extend(format_openai_image_content(text, image_path))
        return messages

    def create_chat_completion(self, messages: list[dict], **kwargs) -> str:
        completion_params = {
            "model": kwargs.get("model", self.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "top_p": kwargs.get("top_p", self.settings.top_p),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "messages": messages,
        }

        completion_func = chatter(self.client)
        return completion_func(self.client, completion_params)

    def create_structured_completion(
        self,
        messages: list[dict],
        response_model: Type[BaseModel],
        **kwargs,
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "top_p": kwargs.get("top_p", self.settings.top_p),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "messages": messages,
        }
        patched_client = self._get_patched_client()
        return patched_client.chat.completions.create(
            response_model=response_model, **completion_params
        )
