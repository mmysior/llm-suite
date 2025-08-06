# 🔧 LLM Suite

A Python package for streamlined interactions with various Large Language Model (LLM) providers, featuring structured output parsing and template-based prompt management.

## Features

- 🤖 **Multi-Provider Support**: Compatible with OpenAI, Anthropic, Ollama, Groq, Perplexity, and LMStudio
- 📋 **Structured Output**: Parse LLM responses directly into Pydantic models
- 📝 **Template Management**: Organize and reuse prompts with Jinja2 templating
- ⚙️ **Configurable**: Easily adjust model parameters and settings

# 🔧 LLM Suite

A Python package for streamlined interactions with various Large Language Model (LLM) providers, featuring structured output parsing and template-based prompt management.

## Features

- 🤖 **Multi-Provider Support**: Compatible with OpenAI, Anthropic, Ollama, Groq, Perplexity, Together, and LMStudio
- 📋 **Structured Output**: Parse LLM responses directly into Pydantic models using Instructor
- 📝 **Template Management**: Organize and reuse prompts with Jinja2 templating
- 🖼️ **Image Support**: Send images to vision-enabled models
- ⚙️ **Configurable**: Easily adjust model parameters and settings
- 🔄 **CLI Tool**: Built-in command line interface for quick interactions

## Getting Started

Install the package using pip:

```bash
pip install llm-suite
```

Alternatively, install from the source:

```bash
git clone https://github.com/mmysior/llm-suite.git
cd llm-suite
pip install -e .
```

## Configuration

Configuration is handled through environment variables. Copy the [`.env.example`](.env.example ) file to [`.env`](.env ) and update with your API keys:

```env
# LLM settings
DEFAULT_PROVIDER="openai"
DEFAULT_MODEL="gpt-4.1-mini"

DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=1.0
DEFAULT_MAX_TOKENS=2048

# Provider-specific settings
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
GROQ_API_KEY=your_groq_key_here
TOGETHER_API_KEY=your_together_key_here
TOGETHER_API_KEY=

OLLAMA_BASE_URL=
LMSTUDIO_BASE_URL=
```


## Usage

### Basic Usage

```python
import llmsuite

# Initialize a chat model
llm = llmsuite.init_chat_model()  # Uses DEFAULT_PROVIDER and DEFAULT_MODEL from .env
# Or specify explicitly
llm = llmsuite.init_chat_model(provider="anthropic", model="claude-3-opus-20240229")

# Simple conversation
messages = llm.build_messages(
    text="Explain quantum computing briefly", 
    system_prompt="You are a helpful assistant."
)
response = llm.chat(messages)
print(response)
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel

class MovieRecommendation(BaseModel):
    title: str
    year: int
    why: str

messages = llm.build_messages(
    text="Recommend a sci-fi movie", 
    system_prompt="You recommend movies."
)
result = llm.extract(
    messages=messages,
    schema=MovieRecommendation
)
print(f"Title: {result.title}, Year: {result.year}")
```

### Working with Images

```python
from pathlib import Path

# Send an image to a vision model
messages = llm.build_messages(
    text="What's in this image?",
    image_path=Path("path/to/image.jpg")
)
response = llm.chat(messages)
```

### Prompt Templates

```python
from llmsuite import get_prompt

# Get a prompt template from your templates directory
prompt = get_prompt("my_template")

# Render the template with variables
rendered_prompt = prompt.compile(variable1="value1", variable2="value2")

# Use the rendered prompt
messages = llm.build_messages(text="Query", system_prompt=rendered_prompt)
response = llm.chat(messages)
```

## Creating Prompt Templates

Create Jinja2 templates with YAML frontmatter metadata:

```jinja
---
type: system
version: 1
author: Your Name
labels: 
    - classification
    - sentiment
tags:
    - example
config:
    temperature: 0.1
    model: gpt-4
---
You are a sentiment analyzer that classifies text as positive, negative, or neutral.

Please analyze the following text:
{{ text }}
```

## Command Line Interface

LLM Suite includes a CLI tool:

```bash
# Simple chat
llmsuite chat "What is the capital of France?"

# Using specific model and provider
llmsuite chat "Explain relativity" --model gpt-4 --provider openai --temperature 0.2

# View current configuration
llmsuite config
```

## Advanced Configuration

For more control, you can customize model parameters:

```python
response = llm.chat(
    messages=messages,
    temperature=0.2,
    max_tokens=500
)
```

## License

MIT
