from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from llmsuite.services.llm_service import LLMService, chatter


def test_chatter_openai():
    """Test chatter function with OpenAI client."""
    mock_client = MagicMock(spec=OpenAI)
    mock_chat = MagicMock()
    mock_completions = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "OpenAI response"

    # Set up the chain of mocks
    mock_client.chat = mock_chat
    mock_chat.completions = mock_completions
    mock_completions.create.return_value = mock_completion

    completion_func = chatter(mock_client)
    result = completion_func(
        mock_client, {"messages": [{"role": "user", "content": "Hello"}]}
    )

    assert result == "OpenAI response"
    mock_completions.create.assert_called_once()


def test_chatter_anthropic():
    """Test chatter function with Anthropic client."""
    mock_client = MagicMock(spec=Anthropic)
    mock_messages = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Anthropic response"
    mock_completion = MagicMock()
    mock_completion.content = [mock_content]

    # Set up the chain of mocks
    mock_client.messages = mock_messages
    mock_messages.create.return_value = mock_completion

    completion_func = chatter(mock_client)
    result = completion_func(
        mock_client, {"messages": [{"role": "user", "content": "Hello"}]}
    )

    assert result == "Anthropic response"
    mock_messages.create.assert_called_once()


class TestLLMService:
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock()
        settings.openai.api_key = "test-key"
        settings.openai.default_model = "gpt-4"
        settings.openai.temperature = 0.7
        settings.openai.top_p = 1.0
        settings.openai.max_tokens = 1000

        settings.anthropic.api_key = "test-key"
        settings.anthropic.default_model = "claude-3-opus"
        settings.anthropic.temperature = 0.7
        settings.anthropic.top_p = 1.0
        settings.anthropic.max_tokens = 1000
        return settings

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        mock = MagicMock(spec=OpenAI)
        return mock

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client."""
        mock = MagicMock(spec=Anthropic)
        return mock

    @patch("llmsuite.services.llm_service.get_settings")
    def test_init_openai(self, mock_get_settings, mock_settings):
        """Test initializing with OpenAI provider."""
        mock_get_settings.return_value = mock_settings

        with patch("llmsuite.services.llm_service.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            service = LLMService("openai")

            assert service.provider == "openai"
            assert service.default_model == "gpt-4"
            mock_openai.assert_called_once_with(api_key="test-key")

    @patch("llmsuite.services.llm_service.get_settings")
    def test_init_anthropic(self, mock_get_settings, mock_settings):
        """Test initializing with Anthropic provider."""
        mock_get_settings.return_value = mock_settings

        with patch("llmsuite.services.llm_service.Anthropic") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()

            service = LLMService("anthropic")

            assert service.provider == "anthropic"
            assert service.default_model == "claude-3-opus"
            mock_anthropic.assert_called_once_with(api_key="test-key")

    @patch("llmsuite.services.llm_service.get_settings")
    def test_init_unsupported_provider(self, mock_get_settings, mock_settings):
        """Test initializing with unsupported provider."""
        mock_get_settings.return_value = mock_settings

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMService("unsupported_provider")

    def test_build_messages_simple(self):
        """Test building simple messages without image or system prompt."""
        with patch.object(LLMService, "_initialize_client"):
            service = LLMService("openai")

            messages = service.build_messages("Hello, world!")

            assert messages == [{"role": "user", "content": "Hello, world!"}]

    def test_build_messages_with_system(self):
        """Test building messages with system prompt."""
        with patch.object(LLMService, "_initialize_client"):
            service = LLMService("openai")

            messages = service.build_messages(
                "Hello, world!", system_prompt="You are a helpful assistant."
            )

            assert messages == [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ]

    @patch("llmsuite.services.llm_service.format_openai_image_content")
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

        with patch.object(LLMService, "_initialize_client"):
            service = LLMService("openai")

            messages = service.build_messages(
                "Describe this image", image_path=Path("test.jpg")
            )

            mock_format_image.assert_called_once_with(
                "Describe this image", Path("test.jpg")
            )
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

    @patch("llmsuite.services.llm_service.format_anthropic_image_content")
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

        with patch.object(LLMService, "_initialize_client"):
            service = LLMService("anthropic")

            messages = service.build_messages(
                "Describe this image", image_path=Path("test.jpg")
            )

            mock_format_image.assert_called_once_with(
                "Describe this image", Path("test.jpg")
            )
            assert messages == [
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

    @patch("llmsuite.services.llm_service.chatter")
    def test_create_chat_completion(self, mock_chatter):
        """Test creating chat completion."""
        mock_completion_func = MagicMock()
        mock_completion_func.return_value = "This is a test response"
        mock_chatter.return_value = mock_completion_func

        with patch.object(LLMService, "_initialize_client"):
            service = LLMService("openai")
            service.default_model = "gpt-4"
            service.settings = MagicMock()
            service.settings.temperature = 0.7
            service.settings.top_p = 1.0
            service.settings.max_tokens = 1000

            messages = [{"role": "user", "content": "Hello"}]
            response = service.create_chat_completion(messages)

            assert response == "This is a test response"
            mock_chatter.assert_called_once_with(service.client)
            mock_completion_func.assert_called_once()

            # Check completion parameters
            call_args = mock_completion_func.call_args[0][1]
            assert call_args["model"] == "gpt-4"
            assert call_args["temperature"] == 0.7
            assert call_args["messages"] == messages

    @patch("llmsuite.services.llm_service.instructor")
    def test_create_structured_completion_openai(self, mock_instructor):
        """Test creating structured completion for OpenAI."""

        class TestModel(BaseModel):
            response: str

        mock_patched_client = MagicMock()
        mock_instructor.from_openai.return_value = mock_patched_client

        mock_response = TestModel(response="This is a structured response")
        mock_patched_client.chat.completions.create.return_value = mock_response

        with patch.object(LLMService, "_initialize_client"):
            service = LLMService("openai")
            service.client = MagicMock(spec=OpenAI)
            service.default_model = "gpt-4"
            service.settings = MagicMock()
            service.settings.temperature = 0.7
            service.settings.top_p = 1.0
            service.settings.max_tokens = 1000

            messages = [{"role": "user", "content": "Hello"}]
            response = service.create_structured_completion(messages, TestModel)

            assert response == mock_response
            mock_instructor.from_openai.assert_called_once_with(service.client)
            mock_patched_client.chat.completions.create.assert_called_once()

    @patch("llmsuite.services.llm_service.instructor")
    def test_create_structured_completion_anthropic(self, mock_instructor):
        """Test creating structured completion for Anthropic."""

        class TestModel(BaseModel):
            response: str

        mock_patched_client = MagicMock()
        mock_instructor.from_anthropic.return_value = mock_patched_client

        mock_response = TestModel(response="This is a structured response")
        mock_patched_client.chat.completions.create.return_value = mock_response

        with patch.object(LLMService, "_initialize_client"):
            service = LLMService("anthropic")
            service.client = MagicMock(spec=Anthropic)
            service.default_model = "claude-3-opus"
            service.settings = MagicMock()
            service.settings.temperature = 0.7
            service.settings.top_p = 1.0
            service.settings.max_tokens = 1000

            messages = [{"role": "user", "content": "Hello"}]
            response = service.create_structured_completion(messages, TestModel)

            assert response == mock_response
            mock_instructor.from_anthropic.assert_called_once_with(service.client)
            mock_patched_client.chat.completions.create.assert_called_once()

    def test_get_patched_client_unsupported(self):
        """Test getting patched client with unsupported client type."""
        with patch.object(LLMService, "_initialize_client"):
            service = LLMService("openai")
            service.client = MagicMock()  # Not OpenAI or Anthropic

            with pytest.raises(ValueError, match="Unsupported client for patching"):
                service._get_patched_client()
