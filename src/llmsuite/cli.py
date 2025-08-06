import logging
from typing import Optional

import click

from .llm import init_chat_model
from .settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool) -> None:
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")


@main.command()
@click.argument("message", required=True)
@click.option(
    "--model", "-m", help="Model to use (e.g., gpt-4, claude-3-sonnet-20240229)", required=False
)
@click.option(
    "--provider",
    "-p",
    help="Provider to use (openai, anthropic, ollama, groq, etc.)",
    required=False,
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    help="Temperature for response generation (0.0 to 2.0)",
    required=False,
)
@click.option("--max-tokens", type=int, help="Maximum number of tokens to generate", required=False)
@click.option("--system-prompt", "-s", help="System prompt to use", required=False)
def chat(
    message: str,
    model: Optional[str],
    provider: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    system_prompt: Optional[str],
) -> None:
    try:
        chat_model = init_chat_model(model=model, provider=provider)
        messages = chat_model.build_messages(text=message, system_prompt=system_prompt)

        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = chat_model.chat(messages, **kwargs)
        click.echo(response)

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@main.command()
def config() -> None:
    """Show current configuration."""
    settings = get_settings()

    # Display configuration with styled output
    click.secho("\n✨ Current Default Settings ✨", fg="green", bold=True)

    # Core defaults section
    click.secho("\n🔧 Core Defaults:", fg="blue")
    click.echo(f"  • Provider: {settings.default_provider or 'Not set'}")
    click.echo(f"  • Model:    {settings.default_model or 'Not set'}")

    # Provider-specific settings
    provider_settings = getattr(settings, settings.default_provider or "openai")
    click.secho(
        f"\n🔧 {settings.default_provider.title() if settings.default_provider else 'OpenAI'} Settings:",
        fg="blue",
    )
    click.echo(f"  • Temperature: {provider_settings.temperature}")
    click.echo(f"  • Max tokens:  {provider_settings.max_tokens}")
    click.echo(f"  • Top P:       {provider_settings.top_p}")

    click.echo()  # Add blank line at the end for cleaner display


if __name__ == "__main__":
    main()
