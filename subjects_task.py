from __future__ import annotations

import random
from typing import List, Optional

import typer
from dotenv import load_dotenv

from prompts import PromptVariant
from utils import TaskExecutor, TaskType, iter_languages, load_model_settings, setup_logging

app = typer.Typer(add_completion=False, help="Run subject vocabulary experiments.")


@app.command()
def run(
    model_family: Optional[List[str]] = typer.Option(
        None,
        "--model",
        "-m",
        help="Limit to one or more model families (e.g., chatgpt, gemini).",
    ),
    model_version: Optional[List[str]] = typer.Option(
        None,
        "--model-version",
        help="Limit to specific version keys defined in config/models.yaml (e.g., gpt4o-mini).",
    ),
    workers: Optional[int] = typer.Option(None, "--workers", help="Number of concurrent prompts to run."),
    language: Optional[List[str]] = typer.Option(None, "--language", "-l", help="Restrict to zh/en."),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility."),
    dry_run: bool = typer.Option(False, help="Print prompts without invoking any model."),
    debug: bool = typer.Option(False, help="Verbose logging of responses."),
) -> None:
    """Execute the subject vocabulary pipeline."""
    load_dotenv()
    setup_logging(verbose=debug)
    if seed is not None:
        random.seed(seed)

    defaults, models = load_model_settings()
    if workers:
        defaults["worker_count"] = workers
    executor = TaskExecutor(TaskType.SUBJECTS, defaults, models)

    languages = tuple(language) if language else tuple(iter_languages())
    executor.run(
        prompt_variants=(PromptVariant.MATCHING, PromptVariant.PROBABILITY),
        languages=languages,
        model_families=model_family,
        model_versions=model_version,
        dry_run=dry_run,
        debug=debug,
    )


if __name__ == "__main__":
    app()
