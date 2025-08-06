import os
from typing import Any, Dict, List, Literal, Optional, cast

import frontmatter
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field

from .settings import get_settings

# ------------------------------------------
# Pydantic Models
# ------------------------------------------


class TemplateMetadata(BaseModel):
    type: Literal["system", "developer", "user"] = "system"
    author: str = ""
    version: int = 1
    labels: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    json_schema: Optional[Dict[str, Any]] = None


class PromptModel(TemplateMetadata):
    name: str
    prompt: str

    def compile(self, **kwargs) -> str:
        env = Environment(
            undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True
        )
        template = env.from_string(self.prompt)
        content = template.render(**kwargs)
        return content


class TemplateSource(BaseModel):
    content: str
    metadata: TemplateMetadata


# ------------------------------------------
# Jinja Environment and Template Loading
# ------------------------------------------


def get_env(templates_dir: Optional[str] = None) -> Environment:
    if templates_dir is None:
        settings = get_settings()
        templates_dir = settings.prompt.templates_dir
    os.makedirs(templates_dir, exist_ok=True)
    return Environment(
        loader=FileSystemLoader(templates_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_template_source(template_path: str, env: Environment) -> TemplateSource:
    """Load and parse a template file with frontmatter metadata."""
    try:
        if env.loader is None:
            raise FileNotFoundError(
                f"No template loader configured for template: {template_path}"
            )

        template_source, _, _ = env.loader.get_source(env, template_path)
        post = frontmatter.loads(template_source)
        metadata = cast(Dict[str, Any], post.metadata)

        return TemplateSource(
            content=post.content,
            metadata=TemplateMetadata(
                **{
                    "type": metadata.get("type", "system"),
                    "author": metadata.get("author", ""),
                    "version": int(metadata.get("version", 1)),
                    "labels": metadata.get("labels", []),
                    "tags": metadata.get("tags", []),
                    "config": metadata.get("config", {}),
                }
            ),
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Template file not found: {template_path}. Error: {str(e)}"
        ) from e


def get_prompt(name: str, templates_dir: Optional[str] = None) -> PromptModel:
    env = get_env(templates_dir)
    template = load_template_source(f"{name}.j2", env)

    return PromptModel(
        name=name,
        prompt=template.content,
        **template.metadata.model_dump(),
    )
