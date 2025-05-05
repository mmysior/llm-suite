import os
import shutil
import tempfile

import pytest

from llmsuite.prompts.prompt_manager import PromptManager


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
def project_template_dir(temp_prompts_dir):
    """Copy the project's actual template.j2 to the temp directory."""
    # Get the path to the project's prompts directory
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    project_template_path = os.path.join(project_dir, "prompts", "template.j2")

    # Copy the template to the temp directory if it exists
    if os.path.exists(project_template_path):
        with open(project_template_path, "r") as src:
            template_content = src.read()
            with open(
                os.path.join(temp_prompts_dir, "project_template.j2"), "w"
            ) as dst:
                dst.write(template_content)

    return temp_prompts_dir


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


class TestPromptManager:
    def test_template_filling(self, temp_prompts_dir):
        """Test if the class correctly fills up the template with variables."""
        # Setup
        manager = PromptManager(templates_dir=temp_prompts_dir)

        # Get prompt and compile it
        prompt = manager.get_prompt("test_template")
        filled_prompt = prompt.compile(name="John")

        # Assert
        assert "Hello, John! This is a test prompt." in filled_prompt

    def test_frontmatter_parsing(self, temp_prompts_dir):
        """Test if the prompt manager correctly parses YAML frontmatter data."""
        # Setup
        manager = PromptManager(templates_dir=temp_prompts_dir)

        # Get prompt with metadata
        prompt = manager.get_prompt("test_template")

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
        """Test if the prompt manager properly shows the raw prompt."""
        # Setup
        manager = PromptManager(templates_dir=temp_prompts_dir)

        # Get prompt
        prompt = manager.get_prompt("test_template")

        # Expected content - the text after the frontmatter section
        expected_content = "Hello, {{ name }}! This is a test prompt."

        # Assert raw prompt content (strip to handle any potential whitespace)
        assert prompt.prompt.strip() == expected_content.strip()

    def test_templates_listing(self, temp_prompts_dir):
        """Test if the prompt manager correctly lists available templates."""
        # Setup
        manager = PromptManager(templates_dir=temp_prompts_dir)

        # Assert
        assert "test_template" in manager.templates

    def test_create_prompt(self, temp_prompts_dir):
        """Test if the prompt manager can create a new prompt."""
        # Setup
        manager = PromptManager(templates_dir=temp_prompts_dir)

        # Create a new prompt
        prompt_content = "This is a {{ descriptor }} prompt."
        manager.create_prompt(
            name="new_template",
            prompt_content=prompt_content,
            type="user",
            author="Test Creator",
            labels=["new"],
            tags=["dynamic"],
            config={"temperature": 0.5},
        )

        # Get the created prompt
        prompt = manager.get_prompt("new_template")

        # Assert
        assert prompt.name == "new_template"
        assert prompt.prompt.strip() == prompt_content.strip()
        assert prompt.type == "user"
        assert prompt.author == "Test Creator"
        assert "new" in prompt.labels
        assert "dynamic" in prompt.tags
        assert prompt.config["temperature"] == 0.5

        # Test filling the newly created template
        filled_prompt = prompt.compile(descriptor="dynamic")
        assert filled_prompt == "This is a dynamic prompt."

    def test_project_template(self, project_template_dir):
        """Test with the actual project template.j2 file."""
        # Setup
        manager = PromptManager(templates_dir=project_template_dir)

        # Get the project template
        prompt = manager.get_prompt("project_template")

        # Assert frontmatter data was parsed correctly
        assert prompt.type == "system"
        assert prompt.version == 1
        assert prompt.author == "Marek Piotr Mysior"
        assert "latest" in prompt.labels
        assert "search" in prompt.tags
        assert "web" in prompt.tags
        assert prompt.config["temperature"] == 0.0
        assert prompt.config["model"] == "gpt-4o-mini"

        # Assert raw prompt content
        assert prompt.prompt.strip() == "PROMPT GOES HERE"

        # Test that we can compile it (even though it doesn't have variables)
        filled_prompt = prompt.compile()
        assert filled_prompt == "PROMPT GOES HERE"

    def test_extended_template_with_variables(self, extended_template_dir):
        """Test an extended version of the template with Jinja variables and control structures."""
        # Setup
        manager = PromptManager(templates_dir=extended_template_dir)

        # Get the extended template
        prompt = manager.get_prompt("extended_template")

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
        assert (
            "Your task is to answer questions for the user named Alice."
            in filled_prompt
        )
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
