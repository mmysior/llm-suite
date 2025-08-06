# 🔧 LLM Suite

A Python package for streamlined interactions with various Large Language Model (LLM) providers, featuring structured output parsing and template-based prompt management.

## Features

- 🤖 **Multi-Provider Support**: Compatible with OpenAI, Anthropic, Ollama, Groq, Perplexity, and LMStudio
- 📋 **Structured Output**: Parse LLM responses directly into Pydantic models
- 📝 **Template Management**: Organize and reuse prompts with Jinja2 templating
- ⚙️ **Configurable**: Easily adjust model parameters and settings

## Getting Started

Install the package using pip

```bash
pip install llm-suite
```

Alternatively, install from the source

```bash
git clone https://github.com/mmysior/llm-suite.git
cd llm-suite
pip install -e .
```

## Configuration

Configuration is handled through environment variables. Update your `.env` file with your API keys:

```env
TEMPLATES_DIR="./prompts"

# LLM settings
DEFAULT_PROVIDER="openai"
DEFAULT_MODEL="gpt-4.1-mini"

DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=1.0
DEFAULT_MAX_TOKENS=2048

OPENAI_API_KEY=
ANTHROPIC_API_KEY=
PERPLEXITY_API_KEY=
GROQ_API_KEY=
TOGETHER_API_KEY=

OLLAMA_BASE_URL=
LMSTUDIO_BASE_URL=
```

Available configuration options:
- `TEMPLATES_DIR` - Custom directory for prompt templates (defaults to package's templates directory).
- `DEFAULT_PROVIDER` - A provider used if no other is explicitly defined at LLMSuite initialization.
- `DEFAULT MODEL` - A default model to use when calling methods on LLMSuite class.

## Usage

### LLM Service

```python
from llmsuite import LLMSuite

# Initialize with your preferred provider (the value will be loaded from the .env file)
llm = LLMSuite() # Loads the provider as specified in the .env file

# Simple chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing briefly."}
]
response = llm.create_chat_completion(messages) # Uses the default model specified in .env file
print(response)

# Structured output with Pydantic
from pydantic import BaseModel

class MovieRecommendation(BaseModel):
    title: str
    year: int
    why: str

messages = [
    {"role": "system", "content": "You recommend movies."},
    {"role": "user", "content": "Recommend a sci-fi movie."}
]
result = llm.create_structured_completion(
    messages=messages,
    response_model=MovieRecommendation
)
print(f"Title: {result.title}, Year: {result.year}")
```

### Prompt Manager

```python
from llmsuite import get_prompt

# Get a prompt template
prompt get_prompt("my_template")

# Render the template with variables
rendered_prompt = prompt.compile(variable1="value1", variable2="value2")

# Use the rendered prompt with LLM service
messages = [{"role": prompt.type, "content": rendered_prompt}]
llm = LLMService(provider="openai")
response = llm.create_chat_completion(messages)
```

## Creating Prompt Templates

Create Jinja2 templates with frontmatter metadata:

```jinja
---
type: system
version: 1
labels: 
    - "classification",
    - "sentiment"
tags:
    - "example"
---
You are a sentiment analyzer that classifies text as positive, negative, or neutral.

Please analyze the following text:
{{ text }}
```

## Advanced Configuration

For more control, you can customize model parameters:

```python
response = llm.create_chat_completion(
    messages=messages,
    model="gpt-4",
    temperature=0.2,
    max_tokens=500
)
```

## License

MIT
