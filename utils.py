from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml
from tqdm import tqdm

from clients import ModelConfig, build_client
from prompts import (
    LANGUAGE_NAMES,
    PromptVariant,
    TaskType,
    build_prompt,
    get_dataset,
)

logger = logging.getLogger("ai_education")


def setup_logging(verbose: bool = False, log_path: Path | str = Path("logs/runner.log")) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def load_model_settings(config_path: Path | str = Path("config/models.yaml")) -> tuple[Dict, List[ModelConfig]]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration file: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    models_cfg = raw.get("models", [])
    models: List[ModelConfig] = []
    for entry in models_cfg:
        family = entry["name"]
        base_display = entry.get("display_name", family)
        versions = entry.get("versions")
        if not versions:
            versions = [
                {
                    "key": "default",
                    "display_name": base_display,
                    "model_id": entry["model_id"],
                }
            ]
        for version in versions:
            version_key = version.get("key", "default")
            merged = {**entry, **version}
            extra_keys = {
                key: value
                for key, value in version.items()
                if key
                not in {
                    "key",
                    "display_name",
                    "model_id",
                    "base_url",
                    "token_url",
                    "env_secret",
                }
            }
            name = f"{family}-{version_key}".lower()
            display_name = version.get("display_name") or f"{base_display} ({version_key})"
            models.append(
                ModelConfig(
                    name=name,
                    display_name=display_name,
                    provider=merged["provider"],
                    model_id=merged["model_id"],
                    env=merged["env"],
                    env_secret=merged.get("env_secret"),
                    base_url=merged.get("base_url"),
                    token_url=merged.get("token_url"),
                    family=family,
                    version=version_key,
                    version_label=version.get("display_name"),
                    extra=extra_keys,
                )
            )
    return defaults, models


def randomize_terms(words: Sequence[str]) -> List[str]:
    shuffled = list(words)
    random.shuffle(shuffled)
    return shuffled


def generate_name_orders(language: str, runs_per_order: int) -> List[Tuple[str, str]]:
    if language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported language: {language}")
    names = LANGUAGE_NAMES[language]
    orders: List[Tuple[str, str]] = []
    for _ in range(runs_per_order):
        orders.append(names)
        orders.append((names[1], names[0]))
    random.shuffle(orders)
    return orders


def clone_dataset_with_words(dataset, words: Sequence[str]):
    return replace(dataset, words=list(words))


class ResultWriter:
    def __init__(self, base_dir: Path | str = Path("data/results"), overwrite: bool = False):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite
        self._cleaned: set[Path] = set()

    def _path(
        self,
        model: ModelConfig,
        task: TaskType,
        language: str,
        variant: PromptVariant,
    ) -> Path:
        family = (model.family or model.name).lower()
        version = (model.version or "default").lower()
        subdir = self.base_dir / task.value / family / version / language / variant.value
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / "standard.jsonl"
        if self.overwrite and path not in self._cleaned:
            path.write_text("", encoding="utf-8")
            self._cleaned.add(path)
        return path

    def append(
        self,
        model: ModelConfig,
        task: TaskType,
        language: str,
        variant: PromptVariant,
        payload: Dict,
    ) -> None:
        path = self._path(model, task, language, variant)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_languages() -> Iterable[str]:
    return ("zh", "en")


def build_prompt_text(
    task: TaskType,
    variant: PromptVariant,
    language: str,
    name_order: Tuple[str, str],
) -> Tuple[str, Sequence[str]]:
    dataset = get_dataset(task, variant, language)
    randomized_words = randomize_terms(dataset.words)
    randomized = clone_dataset_with_words(dataset, randomized_words)
    prompt_body = build_prompt(randomized, name_order)
    return prompt_body, randomized.words


def build_client_for_model(model: ModelConfig):
    return build_client(model)


class TaskExecutor:
    def __init__(self, task: TaskType, defaults: Dict, models: List[ModelConfig]):
        self.task = task
        self.defaults = defaults
        self.models = models
        self.writer = ResultWriter(overwrite=True)
        self.temperature = float(defaults.get("temperature", 0.8))
        self.max_tokens = int(defaults.get("max_output_tokens", 800))
        self.runs_per_order = int(defaults.get("runs_per_name_order", 5))
        self.worker_count = int(defaults.get("worker_count", 1))
        self.max_attempts = int(defaults.get("max_attempts", 3))
        self.retry_backoff_seconds = float(defaults.get("retry_backoff_seconds", 1.0))
        self.retry_backoff_max = float(defaults.get("retry_backoff_max", 8.0))

    def run(
        self,
        prompt_variants: Iterable[PromptVariant],
        languages: Iterable[str],
        model_families: Sequence[str] | None = None,
        model_versions: Sequence[str] | None = None,
        dry_run: bool = False,
        debug: bool = False,
    ) -> None:
        family_filter = {name.lower() for name in model_families} if model_families else None
        version_filter = {name.lower() for name in model_versions} if model_versions else None
        selected_models: List[ModelConfig] = []
        for model in self.models:
            family_key = (model.family or model.name).lower()
            version_key = (model.version or "default").lower()
            if family_filter and family_key not in family_filter:
                continue
            if version_filter and version_key not in version_filter:
                continue
            selected_models.append(model)

        languages = tuple(languages)
        prompt_variants = tuple(prompt_variants)
        orders_per_language = self.runs_per_order * 2
        total_steps = (
            len(selected_models)
            * len(languages)
            * len(prompt_variants)
            * orders_per_language
        )

        progress = tqdm(total=total_steps, desc=f"{self.task.value} runs", unit="prompt")
        try:
            for model in selected_models:
                logger.info("Running task %s for model %s", self.task.value, model.display_name)
                client = build_client_for_model(model)
                for language in languages:
                    orders = generate_name_orders(language, self.runs_per_order)
                    for variant in prompt_variants:
                        self._run_variant(
                            client,
                            model,
                            language,
                            variant,
                            orders,
                            dry_run=dry_run,
                            debug=debug,
                            progress=progress,
                        )
        finally:
            progress.close()

    def _run_variant(
        self,
        client,
        model: ModelConfig,
        language: str,
        variant: PromptVariant,
        orders: List[Tuple[str, str]],
        dry_run: bool,
        debug: bool,
        progress: tqdm | None = None,
    ) -> None:
        jobs = []
        for idx, names in enumerate(orders, start=1):
            prompt_text, shuffled_words = build_prompt_text(
                self.task,
                variant,
                language,
                names,
            )
            meta = {
                "timestamp": timestamp(),
                "model": model.display_name,
                "model_key": model.name,
                "provider": model.provider,
                "model_family": model.family,
                "model_version": model.version,
                "task": self.task.value,
                "variant": variant.value,
                "language": language,
                "name_order": names,
                "words": shuffled_words,
                "prompt_preview": prompt_text[:1200],
                "run_index": idx,
            }
            jobs.append(
                {
                    "prompt": prompt_text,
                    "meta": meta,
                    "language": language,
                    "variant": variant,
                    "model_name": model.name,
                }
            )

        if dry_run:
            for job in jobs:
                snippet = job["prompt"] if debug else job["prompt"][:400]
                logger.info("Dry-run prompt (%s/%s): %s", job["meta"]["model_key"], language, snippet)
                if progress:
                    progress.update(1)
            return

        if self.worker_count <= 1:
            for job in jobs:
                meta_with_result = self._process_job(client, job, debug)
                self.writer.append(
                    model,
                    self.task,
                    job["language"],
                    job["variant"],
                    meta_with_result,
                )
                if progress:
                    progress.update(1)
            return

        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            future_map = {
                executor.submit(self._process_job, client, job, debug): job for job in jobs
            }
            for future in as_completed(future_map):
                job = future_map[future]
                try:
                    meta_with_result = future.result()
                except Exception as exc:  # Should not happen due to internal handling
                    meta_with_result = job["meta"]
                    meta_with_result["error"] = str(exc)
                    logger.error("Model %s failed unexpectedly: %s", job["model_name"], exc)
                self.writer.append(
                    model,
                    self.task,
                    job["language"],
                    job["variant"],
                    meta_with_result,
                )
                if progress:
                    progress.update(1)

    def _send_with_retry(self, client, prompt_text: str) -> str:
        attempt = 0
        delay = self.retry_backoff_seconds
        while attempt < self.max_attempts:
            attempt += 1
            try:
                return client.send(prompt_text, temperature=self.temperature, max_tokens=self.max_tokens)
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_attempts:
                    raise
                logger.warning(
                    "Attempt %s/%s failed (%s). Retrying after %.1fs",
                    attempt,
                    self.max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, self.retry_backoff_max)

    def _process_job(self, client, job: Dict, debug: bool) -> Dict:
        meta = dict(job["meta"])
        try:
            response = self._send_with_retry(client, job["prompt"])
            meta["response"] = response
            if debug:
                logger.debug("Response from %s: %s", job["model_name"], response)
        except Exception as exc:  # noqa: BLE001
            meta["error"] = str(exc)
            logger.error("Model %s failed: %s", job["model_name"], exc)
        return meta
