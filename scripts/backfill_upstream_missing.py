from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from clients import ModelConfig, build_client
from upstream_task import PromptVariant, TaskType, build_prompt, localize_word
from utils import load_model_settings, timestamp


@dataclass
class MissingJob:
    path: Path
    line_index: int  # 0-based
    record: Dict
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun missing upstream entries (empty response / error) and patch JSONL in place."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/results_upstream"),
        help="Root directory of upstream results.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API calls.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N missing jobs (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Only print missing jobs, do not call model APIs.")
    parser.add_argument(
        "--include-error-only",
        action="store_true",
        help="Only backfill rows that have non-empty `error` field.",
    )
    return parser.parse_args()


def collect_missing_jobs(root: Path, error_only: bool = False) -> List[MissingJob]:
    jobs: List[MissingJob] = []
    for path in root.rglob("standard.jsonl"):
        with path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                response = str(rec.get("response") or "").strip()
                error = str(rec.get("error") or "").strip()
                if error_only:
                    if error:
                        jobs.append(MissingJob(path=path, line_index=idx, record=rec, reason="error"))
                    continue
                if (not response) or error:
                    reason = "empty_response" if not response else "error"
                    jobs.append(MissingJob(path=path, line_index=idx, record=rec, reason=reason))
    return jobs


def send_with_retry(
    client,
    prompt_text: str,
    temperature: float,
    max_tokens: int,
    max_attempts: int,
    backoff_seconds: float,
    backoff_max: float,
) -> str:
    attempt = 0
    delay = backoff_seconds
    while attempt < max_attempts:
        attempt += 1
        try:
            return client.send(prompt_text, temperature=temperature, max_tokens=max_tokens)
        except Exception:
            if attempt >= max_attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, backoff_max)
    raise RuntimeError("Exceeded retry attempts")


def resolve_model(
    rec: Dict,
    path: Path,
    by_name: Dict[str, ModelConfig],
    by_family_version: Dict[Tuple[str, str], ModelConfig],
) -> ModelConfig:
    model_key = str(rec.get("model_key") or "").lower()
    if model_key and model_key in by_name:
        return by_name[model_key]

    parts = path.parts
    # data/results_upstream/<task>/<family>/<version>/...
    if "results_upstream" in parts:
        idx = parts.index("results_upstream")
        if len(parts) > idx + 3:
            fam = parts[idx + 2].lower()
            ver = parts[idx + 3].lower()
            model = by_family_version.get((fam, ver))
            if model:
                return model

    raise KeyError(f"Unable to resolve model for record in {path}")


def build_prompt_for_record(rec: Dict) -> str:
    task = TaskType(str(rec["task"]))
    variant = PromptVariant(str(rec["variant"]))
    language = str(rec["language"])
    word = localize_word(task, variant, language, str(rec["word"]))

    order = rec.get("name_order")
    if isinstance(order, list) and len(order) == 2:
        names = (str(order[0]), str(order[1]))
    elif isinstance(order, tuple) and len(order) == 2:
        names = (str(order[0]), str(order[1]))
    else:
        raise ValueError(f"Invalid name_order in record: {order!r}")
    return build_prompt(task, variant, language, word, names)


def patch_files(updates: Dict[Path, Dict[int, Dict]]) -> int:
    patched_rows = 0
    for path, line_updates in updates.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, new_rec in line_updates.items():
            if idx < 0 or idx >= len(lines):
                continue
            lines[idx] = json.dumps(new_rec, ensure_ascii=False)
            patched_rows += 1
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return patched_rows


def main() -> None:
    args = parse_args()
    load_dotenv()

    defaults, models = load_model_settings()
    by_name = {m.name.lower(): m for m in models}
    by_family_version = {
        ((m.family or m.name).lower(), (m.version or "default").lower()): m for m in models
    }

    jobs = collect_missing_jobs(args.root, error_only=args.include_error_only)
    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]

    print(f"Found missing upstream rows: {len(jobs)}")
    if not jobs:
        return

    if args.dry_run:
        for job in jobs[:50]:
            rec = job.record
            print(
                f"{job.path}:{job.line_index + 1} | "
                f"{rec.get('task')}/{rec.get('variant')}/{rec.get('language')} "
                f"| word={rec.get('word')} | reason={job.reason}"
            )
        if len(jobs) > 50:
            print(f"... ({len(jobs) - 50} more)")
        return

    temperature = float(defaults.get("temperature", 0.8))
    max_tokens = int(defaults.get("max_output_tokens", 1200))
    max_attempts = int(defaults.get("max_attempts", 3))
    backoff_seconds = float(defaults.get("retry_backoff_seconds", 1.0))
    backoff_max = float(defaults.get("retry_backoff_max", 8.0))

    client_cache: Dict[str, object] = {}
    updates: Dict[Path, Dict[int, Dict]] = {}
    failed: List[Tuple[MissingJob, str]] = []

    def run_job(job: MissingJob) -> Tuple[MissingJob, Dict]:
        rec = dict(job.record)
        model = resolve_model(rec, job.path, by_name, by_family_version)
        client = client_cache.get(model.name.lower())
        if client is None:
            client = build_client(model)
            client_cache[model.name.lower()] = client

        prompt_text = build_prompt_for_record(rec)
        response = send_with_retry(
            client=client,
            prompt_text=prompt_text,
            temperature=temperature,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            backoff_max=backoff_max,
        )
        rec["timestamp"] = timestamp()
        rec["response"] = response
        rec.pop("error", None)
        rec["backfill"] = {
            "timestamp": timestamp(),
            "reason": job.reason,
        }
        return job, rec

    if args.workers <= 1:
        for i, job in enumerate(jobs, start=1):
            try:
                job_done, new_rec = run_job(job)
                updates.setdefault(job_done.path, {})[job_done.line_index] = new_rec
                if i % 20 == 0:
                    print(f"Processed {i}/{len(jobs)}")
            except Exception as exc:  # noqa: BLE001
                failed.append((job, str(exc)))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_job, job): job for job in jobs}
            done = 0
            for future in as_completed(future_map):
                done += 1
                job = future_map[future]
                try:
                    job_done, new_rec = future.result()
                    updates.setdefault(job_done.path, {})[job_done.line_index] = new_rec
                except Exception as exc:  # noqa: BLE001
                    failed.append((job, str(exc)))
                if done % 20 == 0 or done == len(jobs):
                    print(f"Processed {done}/{len(jobs)}")

    patched_rows = patch_files(updates)
    print(f"Patched rows: {patched_rows}")
    print(f"Failed rows: {len(failed)}")
    if failed:
        fail_path = Path("analysis/upstream_backfill_failures.json")
        payload = [
            {
                "path": str(job.path).replace("\\", "/"),
                "line": job.line_index + 1,
                "reason": job.reason,
                "error": err,
                "task": job.record.get("task"),
                "variant": job.record.get("variant"),
                "language": job.record.get("language"),
                "word": job.record.get("word"),
            }
            for job, err in failed
        ]
        fail_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Failure details: {fail_path}")


if __name__ == "__main__":
    main()
