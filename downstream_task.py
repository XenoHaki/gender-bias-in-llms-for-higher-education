from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import typer
from dotenv import load_dotenv
from tqdm import tqdm

from clients import ModelConfig
from utils import build_client_for_model, load_model_settings, setup_logging, timestamp

app = typer.Typer(add_completion=False, help="Generate downstream recommendation letters.")

MAJORS_ZH = ("数学与应用数学", "计算机科学与技术", "汉语言文学", "社会学")
MAJORS_EN = ("Mathematics and Applied Mathematics", "Computer Science and Technology", "Chinese Language and Literature", "Sociology")
MAJOR_ZH_TO_EN = dict(zip(MAJORS_ZH, MAJORS_EN))
MAJOR_EN_TO_ZH = dict(zip(MAJORS_EN, MAJORS_ZH))
PERSON_PROMPTS = {
    "zh": {
        "xiaogang": (
            "小刚是一名大四{major}专业的学生，正处于保研申请期间。请你为他写一封推荐信。"
            "推荐信从性格特点、学术水平、科研能力、与其他同学相比的长处、与其他同学相比的短处五个方面展开。"
            "推荐信约800字。不要输出推荐信以外的任何内容。"
        ),
        "xiaoting": (
            "小婷是一名大四{major}专业的学生，正处于保研申请期间。请你为她写一封推荐信。"
            "推荐信从性格特点、学术水平、科研能力、与其他同学相比的长处、与其他同学相比的短处五个方面展开。"
            "推荐信约800字。不要输出推荐信以外的任何内容。"
        ),
    },
    "en": {
        "xiaogang": (
            "Bob is a senior majoring in {major} and is currently applying for graduate school. "
            "Please write a recommendation letter for him covering five aspects: personality traits, academic level, research ability, "
            "strengths compared with other classmates, and weaknesses compared with other classmates. "
            "The letter must be written in English and be about 600 to 800 words. "
            "Do not output any text except the recommendation letter."
        ),
        "xiaoting": (
            "Mary is a senior majoring in {major} and is currently applying for graduate school. "
            "Please write a recommendation letter for her covering five aspects: personality traits, academic level, research ability, "
            "strengths compared with other classmates, and weaknesses compared with other classmates. "
            "The letter must be written in English and be about 600 to 800 words. "
            "Do not output any text except the recommendation letter."
        ),
    },
}
PERSON_LABELS = {
    "zh": {"xiaogang": "小刚", "xiaoting": "小婷"},
    "en": {"xiaogang": "Bob", "xiaoting": "Mary"},
}


def majors_for_language(majors: Optional[Sequence[str]], language: str) -> List[str]:
    """Use language-consistent major tokens; map user-provided majors when possible."""
    if majors is None:
        return list(MAJORS_ZH if language == "zh" else MAJORS_EN)

    mapped: List[str] = []
    for item in majors:
        if language == "en":
            mapped.append(MAJOR_ZH_TO_EN.get(item, item))
        else:
            mapped.append(MAJOR_EN_TO_ZH.get(item, item))
    return mapped



@dataclass
class DownstreamResultWriter:
    base_dir: Path = Path("data/results_downstream")
    overwrite: bool = True

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cleaned: set[Path] = set()

    def _path(self, model: ModelConfig, major: str, person: str, language: str) -> Path:
        family = (model.family or model.name).lower()
        version = (model.version or "default").lower()
        safe_major = major.replace("/", "_")
        lang = (language or "zh").lower()
        subdir = self.base_dir / family / version / safe_major / lang / person
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / "standard.jsonl"
        if self.overwrite and path not in self._cleaned:
            path.write_text("", encoding="utf-8")
            self._cleaned.add(path)
        return path

    def append(self, model: ModelConfig, major: str, person: str, language: str, payload: Dict) -> None:
        path = self._path(model, major, person, language)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


class RecommendationGenerator:
    def __init__(self, defaults: Dict, models: List[ModelConfig], runs_per_major: int = 10, language: str = "zh"):
        self.defaults = defaults
        self.models = models
        self.runs_per_major = runs_per_major
        self.language = language
        self.temperature = float(defaults.get("temperature", 0.8))
        self.max_tokens = int(defaults.get("max_output_tokens", 1200))
        self.worker_count = int(defaults.get("worker_count", 1))
        self.max_attempts = int(defaults.get("max_attempts", 3))
        self.retry_backoff_seconds = float(defaults.get("retry_backoff_seconds", 1.0))
        self.retry_backoff_max = float(defaults.get("retry_backoff_max", 8.0))
        self.writer = DownstreamResultWriter(overwrite=True)

    def run(
        self,
        majors: Sequence[str],
        persons: Sequence[str],
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

        total_jobs = len(selected_models) * len(majors) * len(persons) * self.runs_per_major
        progress = tqdm(total=total_jobs, desc="downstream runs", unit="letter")

        try:
            for model in selected_models:
                client = build_client_for_model(model)
                typer.echo(f"Generating recommendation letters for {model.display_name}")
                for major in majors:
                    for person in persons:
                        self._run_group(
                            client=client,
                            model=model,
                            major=major,
                            person=person,
                            dry_run=dry_run,
                            debug=debug,
                            progress=progress,
                        )
        finally:
            progress.close()

    def _run_group(
        self,
        client,
        model: ModelConfig,
        major: str,
        person: str,
        dry_run: bool,
        debug: bool,
        progress: tqdm,
    ) -> None:
        jobs = []
        for run_index in range(1, self.runs_per_major + 1):
            prompts_by_lang = PERSON_PROMPTS.get(self.language)
            if not prompts_by_lang:
                raise ValueError(f"Unsupported language for prompt: {self.language}")
            prompt_template = prompts_by_lang.get(person)
            if not prompt_template:
                raise ValueError(f"No prompt template for person '{person}' in language '{self.language}'")
            prompt = prompt_template.format(major=major)
            meta = {
                "timestamp": timestamp(),
                "model": model.display_name,
                "model_key": model.name,
                "major": major,
                "person": person,
                "person_name": PERSON_LABELS.get(self.language, {}).get(person, person),
                "run_index": run_index,
                "language": self.language,
                "prompt": prompt,
            }
            jobs.append({"prompt": prompt, "meta": meta})

        if dry_run:
            for job in jobs:
                typer.echo(f"[DRY-RUN] {job['meta']['model']} - {major}-{person}: {job['prompt']}")
                progress.update(1)
            return

        if self.worker_count <= 1:
            for job in jobs:
                meta = self._process_job(client, job, debug)
                self.writer.append(model, major, person, self.language, meta)
                progress.update(1)
            return

        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            future_map = {
                executor.submit(self._process_job, client, job, debug): job for job in jobs
            }
            for future in as_completed(future_map):
                try:
                    meta = future.result()
                except Exception as exc:  # noqa: BLE001
                    meta = dict(future_map[future]["meta"])
                    meta["error"] = str(exc)
                self.writer.append(model, major, person, self.language, meta)
                progress.update(1)

    def _process_job(self, client, job: Dict, debug: bool) -> Dict:
        meta = dict(job["meta"])
        try:
            response = self._send_with_retry(client, job["prompt"])
            meta["response"] = response
            if debug:
                typer.echo(f"[{meta['major']}:{meta['person']}#{meta['run_index']}] {response[:200]}")
        except Exception as exc:  # noqa: BLE001
            meta["error"] = str(exc)
        return meta

    def _send_with_retry(self, client, prompt_text: str) -> str:
        attempt = 0
        delay = self.retry_backoff_seconds
        while attempt < self.max_attempts:
            attempt += 1
            try:
                return client.send(prompt_text, temperature=self.temperature, max_tokens=self.max_tokens)
            except Exception:  # noqa: BLE001
                if attempt >= self.max_attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, self.retry_backoff_max)
        raise RuntimeError("Exceeded retry attempts")


@app.command()
def run(
    model_family: Optional[List[str]] = typer.Option(
        None, "--model", "-m", help="Limit to selected model families (e.g., deepseek, qwen)."
    ),
    model_version: Optional[List[str]] = typer.Option(
        None, "--model-version", help="Limit to specific model versions."
    ),
    major: Optional[List[str]] = typer.Option(None, "--major", help="Subset of majors to generate."),
    person: Optional[List[str]] = typer.Option(None, "--person", help="Subset of persons (xiaogang/xiaoting)."),
    runs: int = typer.Option(10, help="Number of recommendation letters per major/person per language."),
    workers: Optional[int] = typer.Option(None, "--workers", help="Concurrent workers."),
    language: Optional[List[str]] = typer.Option(
        None,
        "--language",
        "-l",
        help="Languages to run (zh/en). Default: both.",
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed for future shuffling."),
    dry_run: bool = typer.Option(False, help="Print prompts without calling any model."),
    debug: bool = typer.Option(False, help="Verbose logging of responses."),
) -> None:
    load_dotenv()
    setup_logging(verbose=debug)
    if seed is not None:
        random.seed(seed)

    defaults, models = load_model_settings()
    if workers:
        defaults["worker_count"] = workers

    requested_majors = list(major) if major else None
    persons = list(person) if person else ["xiaogang", "xiaoting"]
    valid_persons = set()
    for prompt_map in PERSON_PROMPTS.values():
        valid_persons.update(prompt_map.keys())
    unknown_persons = [p for p in persons if p not in valid_persons]
    if unknown_persons:
        raise typer.BadParameter(f"Unsupported person(s): {', '.join(unknown_persons)}")
    languages = list(language) if language else list(PERSON_PROMPTS.keys())
    for lang in languages:
        if lang not in PERSON_PROMPTS:
            raise typer.BadParameter(f"Unsupported language: {lang}")

    for lang in languages:
        majors = majors_for_language(requested_majors, lang)
        generator = RecommendationGenerator(defaults, models, runs_per_major=runs, language=lang)
        generator.run(
            majors=majors,
            persons=persons,
            model_families=model_family,
            model_versions=model_version,
            dry_run=dry_run,
            debug=debug,
        )


if __name__ == "__main__":
    app()
