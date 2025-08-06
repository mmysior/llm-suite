from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from llmsuite.llm import (
    build_messages,
    chat,
    chatter,
    extract,
    get_client,
    init_chat_model,
)


class TestChatter:
    """Test the chatter function."""

    def test_chatter_openai_success(self):
        """Test chatter function with OpenAI client returns completion."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "OpenAI response"
        mock_client.chat.completions.create.return_value = mock_completion

        # Simulate isinstance check
        with patch("llmsuite.llm.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, cls: cls == OpenAI

            completion_func = chatter(mock_client)
            result = completion_func(
                mock_client, {"messages": [{"role": "user", "content": "Hello"}]}
            )

            assert result == "OpenAI response"
            mock_client.chat.completions.create.assert_called_once_with(
                messages=[{"role": "user", "content": "Hello"}]
            )

    def test_chatter_openai_empty_content(self):
        """Test chatter function with OpenAI client when content is None."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_completion

        # Simulate isinstance check
        with patch("llmsuite.llm.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, cls: cls == OpenAI

            completion_func = chatter(mock_client)
            result = completion_func(
                mock_client, {"messages": [{"role": "user", "content": "Hello"}]}
            )

            assert result == ""

    def test_chatter_openai_exception(self):
        """Test chatter function with OpenAI client when exception occurs."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        # Simulate isinstance check
        with patch("llmsuite.llm.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, cls: cls == OpenAI

            completion_func = chatter(mock_client)

            with pytest.raises(RuntimeError, match="OpenAI completion failed: API Error"):
                completion_func(mock_client, {"messages": [{"role": "user", "content": "Hello"}]})

    def test_chatter_anthropic_success(self):
        """Test chatter function with Anthropic client returns completion."""
        mock_client = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Anthropic response"
        mock_completion = MagicMock()
        mock_completion.content = [mock_content]
        mock_client.messages.create.return_value = mock_completion

        # Simulate isinstance check
        with patch("llmsuite.llm.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, cls: cls == Anthropic

            completion_func = chatter(mock_client)
            result = completion_func(
                mock_client, {"messages": [{"role": "user", "content": "Hello"}]}
            )

            assert result == "Anthropic response"
            mock_client.messages.create.assert_called_once_with(
                messages=[{"role": "user", "content": "Hello"}]
            )

    def test_chatter_anthropic_with_system_message(self):
        """Test chatter function with Anthropic client with system message."""
        mock_client = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Anthropic response"
        mock_completion = MagicMock()
        mock_completion.content = [mock_content]
        mock_client.messages.create.return_value = mock_completion

        # Simulate isinstance check
        with patch("llmsuite.llm.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, cls: cls == Anthropic

            completion_func = chatter(mock_client)
            result = completion_func(
                mock_client,
                {
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hello"},
                    ],
                    "temperature": 0.7,
                },
            )

            assert result == "Anthropic response"
            mock_client.messages.create.assert_called_once_with(
                messages=[{"role": "user", "content": "Hello"}],
                system="You are helpful",
                temperature=0.7,
            )

    def test_chatter_anthropic_exception(self):
        """Test chatter function with Anthropic client when exception occurs."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")

        # Simulate isinstance check
        with patch("llmsuite.llm.isinstance") as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, cls: cls == Anthropic

            completion_func = chatter(mock_client)

            with pytest.raises(RuntimeError, match="Anthropic completion failed: API Error"):
                completion_func(mock_client, {"messages": [{"role": "user", "content": "Hello"}]})

    def test_chatter_unsupported_client(self):
        """Test chatter function with unsupported client type."""
        mock_client = MagicMock()  # Not OpenAI or Anthropic

        with pytest.raises(ValueError, match="Unsupported client type"):
            chatter(mock_client)


class TestGetClient:
    """Test the get_client function."""

    @patch("llmsuite.llm.get_settings")
    @patch("llmsuite.llm.OpenAI")
    def test_get_client_openai(self, mock_openai, mock_get_settings):
        """Test getting OpenAI client."""
        mock_settings = MagicMock()
        mock_settings.openai.api_key = "test-key"
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        result = get_client("openai")

        assert result == mock_client
        mock_openai.assert_called_once_with(api_key="test-key")

    @patch("llmsuite.llm.get_settings")
    @patch("llmsuite.llm.Anthropic")
    def test_get_client_anthropic(self, mock_anthropic, mock_get_settings):
        """Test getting Anthropic client."""
        mock_settings = MagicMock()
        mock_settings.anthropic.api_key = "test-key"
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        result = get_client("anthropic")

        assert result == mock_client
        mock_anthropic.assert_called_once_with(api_key="test-key")

    @patch("llmsuite.llm.get_settings")
    @patch("llmsuite.llm.OpenAI")
    def test_get_client_ollama(self, mock_openai, mock_get_settings):
        """Test getting Ollama client (OpenAI-compatible)."""
        mock_settings = MagicMock()
        mock_settings.ollama.base_url = "http://localhost:11434"
        mock_settings.ollama.api_key = "test-key"
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        result = get_client("ollama")

        assert result == mock_client
        mock_openai.assert_called_once_with(base_url="http://localhost:11434", api_key="test-key")

    @patch("llmsuite.llm.get_settings")
    @patch("llmsuite.llm.OpenAI")
    def test_get_client_groq(self, mock_openai, mock_get_settings):
        """Test getting Groq client (OpenAI-compatible)."""
        mock_settings = MagicMock()
        mock_settings.groq.base_url = "https://api.groq.com/openai/v1"
        mock_settings.groq.api_key = "test-key"
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        result = get_client("groq")

        assert result == mock_client
        mock_openai.assert_called_once_with(
            base_url="https://api.groq.com/openai/v1", api_key="test-key"
        )

    @patch("llmsuite.llm.get_settings")
    def test_get_client_unsupported(self, mock_get_settings):
        """Test getting client for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
            get_client("unsupported")


class TestBuildMessages:
    """Test the build_messages function."""

    def test_build_messages_simple(self):
        """Test building simple messages without image or system prompt."""
        messages = build_messages("openai", "Hello, world!")

        assert messages == [{"role": "user", "content": "Hello, world!"}]

    def test_build_messages_with_system_prompt(self):
        """Test building messages with system prompt."""
        messages = build_messages(
            "openai", "Hello, world!", system_prompt="You are a helpful assistant."
        )

        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ]

    @patch("llmsuite.llm.format_openai_image_content")
    def test_build_messages_with_image_openai(self, mock_format_image):
        """Test building messages with image for OpenAI."""
        mock_format_image.return_value = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,..."},
                    },
                ],
            }
        ]

        messages = build_messages("openai", "Describe this image", image_path=Path("test.jpg"))

        mock_format_image.assert_called_once_with("Describe this image", Path("test.jpg"))
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,..."},
                    },
                ],
            }
        ]

    @patch("llmsuite.llm.format_anthropic_image_content")
    def test_build_messages_with_image_anthropic(self, mock_format_image):
        """Test building messages with image for Anthropic."""
        mock_format_image.return_value = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "...",
                        },
                    },
                ],
            }
        ]

        messages = build_messages(
            "anthropic",
            "Describe this image",
            image_path=Path("test.jpg"),
            system_prompt="You are an image analyzer.",
        )

        mock_format_image.assert_called_once_with("Describe this image", Path("test.jpg"))
        assert messages == [
            {"role": "system", "content": "You are an image analyzer."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "...",
                        },
                    },
                ],
            },
        ]


class TestChat:
    """Test the chat function."""

    @patch("llmsuite.llm.chatter")
    @patch("llmsuite.llm.get_client")
    @patch("llmsuite.llm.get_settings")
    def test_chat_success(self, mock_get_settings, mock_get_client, mock_chatter):
        """Test successful chat completion."""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.openai.temperature = 0.7
        mock_settings.openai.top_p = 1.0
        mock_settings.openai.max_tokens = 1000
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_completion_func = MagicMock()
        mock_completion_func.return_value = "This is a test response"
        mock_chatter.return_value = mock_completion_func

        messages = [{"role": "user", "content": "Hello"}]
        result = chat(messages, "gpt-4", "openai")

        assert result == "This is a test response"
        mock_get_client.assert_called_once_with("openai")
        mock_chatter.assert_called_once_with(mock_client)
        mock_completion_func.assert_called_once_with(
            mock_client,
            {
                "model": "gpt-4",
                "temperature": 0.7,
                "top_p": 1.0,
                "max_tokens": 1000,
                "messages": messages,
            },
        )

    @patch("llmsuite.llm.chatter")
    @patch("llmsuite.llm.get_client")
    @patch("llmsuite.llm.get_settings")
    def test_chat_with_custom_params(self, mock_get_settings, mock_get_client, mock_chatter):
        """Test chat with custom parameters."""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.openai.temperature = 0.7
        mock_settings.openai.top_p = 1.0
        mock_settings.openai.max_tokens = 1000
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_completion_func = MagicMock()
        mock_completion_func.return_value = "Custom response"
        mock_chatter.return_value = mock_completion_func

        messages = [{"role": "user", "content": "Hello"}]
        result = chat(
            messages,
            "gpt-4",
            "openai",
            temperature=0.9,
            max_tokens=500,
            top_p=0.8,
        )

        assert result == "Custom response"
        mock_completion_func.assert_called_once_with(
            mock_client,
            {
                "model": "gpt-4",
                "temperature": 0.9,
                "top_p": 0.8,
                "max_tokens": 500,
                "messages": messages,
            },
        )


class TestExtract:
    """Test the extract function."""

    @patch("llmsuite.llm.instructor")
    @patch("llmsuite.llm.get_client")
    @patch("llmsuite.llm.get_settings")
    @patch("llmsuite.llm.isinstance")
    def test_extract_openai_success(
        self, mock_isinstance, mock_get_settings, mock_get_client, mock_instructor
    ):
        """Test successful extraction with OpenAI client."""

        class TestModel(BaseModel):
            response: str

        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.openai.temperature = 0.7
        mock_settings.openai.top_p = 1.0
        mock_settings.openai.max_tokens = 1000
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_patched_client = MagicMock()
        mock_instructor.from_openai.return_value = mock_patched_client

        mock_response = TestModel(response="Extracted data")
        mock_patched_client.chat.completions.create.return_value = mock_response

        # Simulate isinstance check
        mock_isinstance.side_effect = lambda obj, cls: cls == OpenAI

        messages = [{"role": "user", "content": "Extract data"}]
        result = extract(messages, TestModel, "gpt-4", "openai")

        assert result == mock_response
        mock_get_client.assert_called_once_with("openai")
        mock_instructor.from_openai.assert_called_once_with(mock_client)
        mock_patched_client.chat.completions.create.assert_called_once_with(
            response_model=TestModel,
            model="gpt-4",
            temperature=0.7,
            top_p=1.0,
            max_tokens=1000,
            messages=messages,
        )

    @patch("llmsuite.llm.instructor")
    @patch("llmsuite.llm.get_client")
    @patch("llmsuite.llm.get_settings")
    @patch("llmsuite.llm.isinstance")
    def test_extract_anthropic_success(
        self, mock_isinstance, mock_get_settings, mock_get_client, mock_instructor
    ):
        """Test successful extraction with Anthropic client."""

        class TestModel(BaseModel):
            response: str

        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.anthropic.temperature = 0.7
        mock_settings.anthropic.top_p = 1.0
        mock_settings.anthropic.max_tokens = 1000
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_patched_client = MagicMock()
        mock_instructor.from_anthropic.return_value = mock_patched_client

        mock_response = TestModel(response="Extracted data")
        mock_patched_client.chat.completions.create.return_value = mock_response

        # Simulate isinstance check
        mock_isinstance.side_effect = lambda obj, cls: cls == Anthropic

        messages = [{"role": "user", "content": "Extract data"}]
        result = extract(messages, TestModel, "claude-3-opus", "anthropic")

        assert result == mock_response
        mock_get_client.assert_called_once_with("anthropic")
        mock_instructor.from_anthropic.assert_called_once_with(mock_client)
        mock_patched_client.chat.completions.create.assert_called_once_with(
            response_model=TestModel,
            model="claude-3-opus",
            temperature=0.7,
            top_p=1.0,
            max_tokens=1000,
            messages=messages,
        )

    @patch("llmsuite.llm.get_client")
    @patch("llmsuite.llm.get_settings")
    def test_extract_unsupported_client(self, mock_get_settings, mock_get_client):
        """Test extract with unsupported client type."""

        class TestModel(BaseModel):
            response: str

        # Setup mocks
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()  # Not OpenAI or Anthropic
        mock_get_client.return_value = mock_client

        messages = [{"role": "user", "content": "Extract data"}]

        with pytest.raises(ValueError, match="Unsupported client for patching"):
            extract(messages, TestModel, "gpt-4", "openai")


class TestInitChatModel:
    """Test the init_chat_model function."""

    @patch("llmsuite.llm.get_settings")
    def test_init_chat_model_with_defaults(self, mock_get_settings):
        """Test initializing chat model with default settings."""
        mock_settings = MagicMock()
        mock_settings.default_provider = "openai"
        mock_settings.default_model = "gpt-4"
        mock_get_settings.return_value = mock_settings

        chat_model = init_chat_model()

        assert chat_model._provider == "openai"
        assert chat_model._model == "gpt-4"

    @patch("llmsuite.llm.get_settings")
    def test_init_chat_model_with_custom_params(self, mock_get_settings):
        """Test initializing chat model with custom parameters."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        chat_model = init_chat_model(model="claude-3-opus", provider="anthropic")

        assert chat_model._provider == "anthropic"
        assert chat_model._model == "claude-3-opus"


class TestChatModel:
    """Test the ChatModel class returned by init_chat_model."""

    @patch("llmsuite.llm.build_messages")
    @patch("llmsuite.llm.get_settings")
    def test_chat_model_build_messages(self, mock_get_settings, mock_build_messages):
        """Test ChatModel.build_messages method."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_build_messages.return_value = [{"role": "user", "content": "Hello"}]

        chat_model = init_chat_model(model="gpt-4", provider="openai")
        result = chat_model.build_messages("Hello")

        mock_build_messages.assert_called_once_with("openai", "Hello", None, None)
        assert result == [{"role": "user", "content": "Hello"}]

    @patch("llmsuite.llm.chat")
    @patch("llmsuite.llm.get_settings")
    def test_chat_model_chat(self, mock_get_settings, mock_chat):
        """Test ChatModel.chat method."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_chat.return_value = "Chat response"

        chat_model = init_chat_model(model="gpt-4", provider="openai")
        messages = [{"role": "user", "content": "Hello"}]
        result = chat_model.chat(messages, temperature=0.9)

        mock_chat.assert_called_once_with(messages, "gpt-4", "openai", temperature=0.9)
        assert result == "Chat response"

    @patch("llmsuite.llm.extract")
    @patch("llmsuite.llm.get_settings")
    def test_chat_model_extract(self, mock_get_settings, mock_extract):
        """Test ChatModel.extract method."""

        class TestModel(BaseModel):
            response: str

        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_extract.return_value = TestModel(response="Extracted")

        chat_model = init_chat_model(model="gpt-4", provider="openai")
        messages = [{"role": "user", "content": "Extract"}]
        result = chat_model.extract(messages, TestModel, temperature=0.1)

        mock_extract.assert_called_once_with(
            messages, TestModel, "gpt-4", "openai", temperature=0.1
        )
        assert result.response == "Extracted"
