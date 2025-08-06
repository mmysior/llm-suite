import os
import shutil
import tempfile

import pytest

from llmsuite.prompts import get_prompt


@pytest.fixture
def temp_prompts_dir():
    """Create a temporary directory for prompts and copy the template file."""
    temp_dir = tempfile.mkdtemp()

    # Create a test template in the temporary directory
    template_content = """---
type: system
version: 1
author: Test Author
labels:
    - test
tags:
    - unit
    - test
config: {
    "temperature": 0.7,
    "model": "test-model",
}
---
Hello, {{ name }}! This is a test prompt."""

    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, "test_template.j2"), "w") as f:
        f.write(template_content)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def extended_template_dir(temp_prompts_dir):
    """Create an extended version of template.j2 with variables for testing."""
    extended_template_content = """---
type: system
version: 2
author: Test Author
labels:
    - latest
    - extended
tags:
    - search
    - web
    - testing
config: {
    "temperature": 0.1,
    "model": "gpt-4",
}
---
You are a {{ role }} AI assistant.

Your task is to {{ task }} for the user named {{ user_name }}.

Please follow these instructions:
{% for instruction in instructions %}
- {{ instruction }}
{% endfor %}

Thank you for using our service!"""

    with open(os.path.join(temp_prompts_dir, "extended_template.j2"), "w") as f:
        f.write(extended_template_content)

    return temp_prompts_dir


@pytest.fixture
def project_template_dir():
    """Get the path to the project's actual template.j2 file."""
    # Get the path to the project's prompts directory
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    prompts_dir = os.path.join(project_dir, "prompts")
    return prompts_dir


class TestPrompts:
    def test_template_filling(self, temp_prompts_dir):
        """Test if the function correctly fills up the template with variables."""
        # Get prompt and compile it
        prompt = get_prompt("test_template", templates_dir=temp_prompts_dir)
        filled_prompt = prompt.compile(name="John")

        # Assert
        assert "Hello, John! This is a test prompt." in filled_prompt

    def test_frontmatter_parsing(self, temp_prompts_dir):
        """Test if the prompt system correctly parses YAML frontmatter data."""
        # Get prompt with metadata
        prompt = get_prompt("test_template", templates_dir=temp_prompts_dir)

        # Assert frontmatter data was parsed correctly
        assert prompt.type == "system"
        assert prompt.version == 1
        assert prompt.author == "Test Author"
        assert "test" in prompt.labels
        assert "unit" in prompt.tags
        assert "test" in prompt.tags
        assert prompt.config["temperature"] == 0.7
        assert prompt.config["model"] == "test-model"

    def test_raw_prompt_display(self, temp_prompts_dir):
        """Test if the prompt system properly shows the raw prompt."""
        # Get prompt
        prompt = get_prompt("test_template", templates_dir=temp_prompts_dir)

        # Expected content - the text after the frontmatter section
        expected_content = "Hello, {{ name }}! This is a test prompt."

        # Assert raw prompt content (strip to handle any potential whitespace)
        assert prompt.prompt.strip() == expected_content.strip()

    def test_project_template_evaluation(self, project_template_dir):
        """Test the actual project template.j2 file evaluation and metadata parsing."""
        # Get the project template
        prompt = get_prompt("template", templates_dir=project_template_dir)

        # Assert frontmatter data was parsed correctly
        assert prompt.type == "system"
        assert prompt.version == 1
        assert prompt.author == "Marek Piotr Mysior"
        assert "latest" in prompt.labels
        assert "tag1" in prompt.tags
        assert "tag2" in prompt.tags
        assert prompt.config["temperature"] == 0.1
        assert prompt.config["model"] == "gpt-4.1"

        # Check that json_schema is properly parsed
        assert "json_schema" in prompt.config
        json_schema = prompt.config["json_schema"]
        assert json_schema["name"] == "response_model"
        assert json_schema["strict"] is True
        assert "schema" in json_schema
        schema = json_schema["schema"]
        assert schema["type"] == "object"
        assert "reasoning" in schema["properties"]
        assert "response" in schema["properties"]
        assert schema["required"] == ["reasoning", "response"]

        # Assert raw prompt content includes the expected structure
        assert "[Snippet Activated: Prompt Name]" in prompt.prompt
        assert "<objective>" in prompt.prompt
        assert "<context>" in prompt.prompt
        assert "<rules>" in prompt.prompt
        assert "{{ variable_name }}" in prompt.prompt

        # Test that we can compile it without variables
        filled_prompt = prompt.compile(variable_name="example_value")
        assert "[Snippet Activated: Prompt Name]" in filled_prompt
        assert "A key objective for the LLM" in filled_prompt
        assert "Any information that is relevant to the task" in filled_prompt
        assert "OVERRIDE all default behaviour" in filled_prompt

        # Test that we can compile it with variables (the template mentions {{ variable_name }} format)
        filled_prompt_with_vars = prompt.compile(variable_name="test_value")
        assert "test_value" in filled_prompt_with_vars

    def test_extended_template_with_variables(self, extended_template_dir):
        """Test an extended version of the template with Jinja variables and control structures."""
        # Get the extended template
        prompt = get_prompt("extended_template", templates_dir=extended_template_dir)

        # Assert frontmatter data was parsed correctly
        assert prompt.type == "system"
        assert prompt.version == 2
        assert prompt.author == "Test Author"
        assert "latest" in prompt.labels
        assert "extended" in prompt.labels
        assert "testing" in prompt.tags
        assert prompt.config["temperature"] == 0.1
        assert prompt.config["model"] == "gpt-4"

        # Test with simple variables
        filled_prompt = prompt.compile(
            role="helpful", task="answer questions", user_name="Alice", instructions=[]
        )

        assert "You are a helpful AI assistant." in filled_prompt
        assert "Your task is to answer questions for the user named Alice." in filled_prompt
        assert "Please follow these instructions:" in filled_prompt
        assert "Thank you for using our service!" in filled_prompt

        # Test with a list for the loop
        filled_prompt = prompt.compile(
            role="coding",
            task="write code",
            user_name="Bob",
            instructions=["Use Python", "Add type hints", "Write tests"],
        )

        assert "You are a coding AI assistant." in filled_prompt
        assert "Your task is to write code for the user named Bob." in filled_prompt
        assert "- Use Python" in filled_prompt
        assert "- Add type hints" in filled_prompt
        assert "- Write tests" in filled_prompt
