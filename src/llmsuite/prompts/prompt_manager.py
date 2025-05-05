import os
from typing import Any, Dict, List, Literal, Optional, cast

import frontmatter
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field


class TemplateMetadata(BaseModel):
    type: Literal["system", "developer", "user"] = "system"
    author: str = ""
    version: int = 1
    labels: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


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
# Prompt Manager
# ------------------------------------------


class PromptManager:
    def __init__(self, templates_dir: Optional[str] = None):
        self._templates_dir = self._get_templates_dir(templates_dir)
        self._env = self._get_env()
        self.templates = self._load_templates()

    def _get_templates_dir(self, templates_dir: Optional[str] = None) -> str:
        templates_dir = templates_dir or os.getenv("TEMPLATES_DIR", "./prompts")
        os.makedirs(templates_dir, exist_ok=True)
        return templates_dir

    def _get_env(self) -> Environment:
        return Environment(
            loader=FileSystemLoader(self._templates_dir),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _load_template_source(self, template_path: str) -> TemplateSource:
        try:
            if self._env.loader is None:
                raise FileNotFoundError(
                    f"No template loader configured for template: {template_path}"
                )
            template_source, _, _ = self._env.loader.get_source(
                self._env, template_path
            )

            post = frontmatter.loads(template_source)
            metadata = cast(Dict[str, Any], post.metadata)

            return TemplateSource(
                content=post.content,
                metadata=TemplateMetadata(
                    type=cast(
                        Literal["system", "developer", "user"],
                        metadata.get("type", "system"),
                    ),
                    author=metadata.get("author", ""),
                    version=int(str(metadata.get("version", 1))),
                    labels=cast(List[str], metadata.get("labels", [])),
                    tags=cast(List[str], metadata.get("tags", [])),
                    config=cast(Dict[str, Any], metadata.get("config", {})),
                ),
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Template file not found: {template_path}. Error: {str(e)}"
            ) from e

    def _load_templates(self) -> List[str]:
        return [
            template_path.replace(".j2", "")
            for template_path in os.listdir(self._templates_dir)
            if template_path.endswith(".j2")
        ]

    def get_prompt(self, template_name: str) -> PromptModel:
        template_path = f"{template_name}.j2"
        template = self._load_template_source(template_path)

        return PromptModel(
            name=template_name,
            prompt=template.content,
            **template.metadata.model_dump(),
        )

    def create_prompt(
        self,
        name: str,
        prompt_content: str,
        type: Literal["system", "developer", "user"] = "system",
        author: str = "",
        labels: List[str] = [],
        tags: List[str] = [],
        config: Dict[str, Any] = {},
    ) -> str:
        template_path = os.path.join(self._templates_dir, f"{name}.j2")
        if os.path.exists(template_path):
            raise FileExistsError(f"Template already exists: {template_path}")

        # Prepare metadata
        metadata = {
            "type": type,
            "version": 1,
            "author": author,
            "labels": labels,
            "tags": tags,
            "config": config,
        }

        # Create template content with frontmatter
        post = frontmatter.Post(prompt_content, **metadata)
        template_content = frontmatter.dumps(post)

        # Write to file
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(template_content)

        # Reload the Jinja environment to pick up the new template
        self._env = self._get_env()
        self.templates = self._load_templates()
        return template_path
