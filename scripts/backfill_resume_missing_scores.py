from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from resume_task import build_prompt
from utils import build_client_for_model, load_model_settings, timestamp


SCORING_BASE = Path("data/results_resume/scoring")


@dataclass
class MissingJob:
    path: Path
    line_index: int  # 0-based in non-empty lines list
    record: Dict
    family: str
    version: str
    reason: str


def _parse_score_from_text(text: str) -> Optional[float]:
    content = (text or "").strip()
    if not content:
        return None

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    segments = list(reversed(lines)) if lines else [content]

    def _normalize_numeric(v: float) -> Optional[float]:
        # Keep consistent with export logic: valid score is 10~100.
        if 10 <= v <= 100:
            return v
        return None

    def _parse_single_line_strict(line: str) -> Optional[float]:
        cleaned = line.strip().rstrip("。.!?;；,，")
        pure_digits = re.fullmatch(r"\d+", cleaned)
        if pure_digits:
            raw = pure_digits.group(0)
            raw_int = int(raw)
            normalized = _normalize_numeric(float(raw_int))
            if normalized is not None:
                return normalized
            if len(raw) == 4:
                left = int(raw[:2])
                right = int(raw[2:])
                if 0 <= left <= 100 and 0 <= right <= 100:
                    avg = (left + right) / 2.0
                    normalized = _normalize_numeric(avg)
                    if normalized is not None:
                        return normalized
            for tail_len in (2, 3):
                if len(raw) >= tail_len:
                    tail = int(raw[-tail_len:])
                    normalized = _normalize_numeric(float(tail))
                    if normalized is not None:
                        return normalized

        pure_number = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned)
        if pure_number:
            try:
                value = float(cleaned)
            except ValueError:
                return None
            return _normalize_numeric(value)

        one_number = re.fullmatch(r"[^0-9/%-]*([-+]?\d+(?:\.\d+)?)[^0-9/%]*", cleaned)
        if one_number:
            try:
                value = float(one_number.group(1))
            except ValueError:
                return None
            return _normalize_numeric(value)
        return None

    for line in lines:
        v = _parse_single_line_strict(line)
        if v is not None:
            return v

    for line in reversed(lines):
        v = _parse_single_line_strict(line)
        if v is not None:
            return v

    for segment in segments:
        pure_digits = re.fullmatch(r"\d+", segment)
        if pure_digits:
            raw = pure_digits.group(0)
            raw_int = int(raw)
            normalized = _normalize_numeric(float(raw_int))
            if normalized is not None:
                return normalized
            if len(raw) == 4:
                left = int(raw[:2])
                right = int(raw[2:])
                if 0 <= left <= 100 and 0 <= right <= 100:
                    avg = (left + right) / 2.0
                    normalized = _normalize_numeric(avg)
                    if normalized is not None:
                        return normalized
            for tail_len in (2, 3):
                if len(raw) >= tail_len:
                    tail = int(raw[-tail_len:])
                    normalized = _normalize_numeric(float(tail))
                    if normalized is not None:
                        return normalized

        matches = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", segment))
        for m in reversed(matches):
            start = m.start()
            if segment[:start].rstrip().endswith("/"):
                continue
            try:
                value = float(m.group(0))
            except ValueError:
                continue
            normalized = _normalize_numeric(value)
            if normalized is not None:
                return normalized
    return None


def _effective_score(rec: Dict) -> Optional[float]:
    # Prefer reparsing score_response; old stored `score` may be wrong.
    parsed = _parse_score_from_text(str(rec.get("score_response") or ""))
    if parsed is not None:
        return parsed
    score = rec.get("score")
    if isinstance(score, (int, float)):
        value = float(score)
        if 10 <= value <= 100:
            return value
    return None


def _send_with_retry(client, prompt_text: str, temperature: float, max_tokens: int) -> str:
    retries = 0
    max_retries = 5
    delay = 1.0
    while True:
        try:
            return client.send(prompt_text, temperature=temperature, max_tokens=max_tokens)
        except Exception:
            if retries >= max_retries:
                raise
            time.sleep(delay)
            delay *= 2
            retries += 1


def _resolve_model_from_path(path: Path) -> Tuple[str, str]:
    # data/results_resume/scoring/<family>/<version>/standard.jsonl
    parts = path.parts
    idx = parts.index("scoring")
    return parts[idx + 1].lower(), parts[idx + 2].lower()


def _collect_missing_jobs(
    base: Path,
    family_filter: Optional[set[str]] = None,
    version_filter: Optional[set[str]] = None,
    language_filter: Optional[set[str]] = None,
) -> List[MissingJob]:
    jobs: List[MissingJob] = []
    for path in sorted(base.rglob("standard.jsonl")):
        family, version = _resolve_model_from_path(path)
        if family_filter and family not in family_filter:
            continue
        if version_filter and version not in version_filter:
            continue
        with path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                lang = str(rec.get("language") or "").lower()
                if language_filter and lang not in language_filter:
                    continue
                eff = _effective_score(rec)
                if eff is None:
                    reason = "blank_score_response" if not str(rec.get("score_response") or "").strip() else "invalid_score"
                    jobs.append(
                        MissingJob(
                            path=path,
                            line_index=idx,
                            record=rec,
                            family=family,
                            version=version,
                            reason=reason,
                        )
                    )
    return jobs


def _build_spec_from_record(rec: Dict) -> Dict:
    return {
        "language": rec["language"],
        "person_id": rec["person_id"],
        "name": rec["name"],
        "gender_id": rec.get("gender_id"),
        "gender": rec["gender"],
        "major": rec["major"],
        "gpa": rec["gpa"],
        "competition": rec["competition"],
        "internship": rec["internship"],
        "english": rec["english"],
        "measurement": rec["measurement"],
    }


def _patch_files(updates: Dict[Path, Dict[int, Dict]]) -> int:
    patched = 0
    for path, line_updates in updates.items():
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        for idx, rec in line_updates.items():
            if 0 <= idx < len(raw_lines):
                raw_lines[idx] = json.dumps(rec, ensure_ascii=False)
                patched += 1
        path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    return patched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loop-rerun real-missing resume scores.")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent workers.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum rerun rounds.")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-round limit; 0 means no limit.")
    parser.add_argument("--dry-run", action="store_true", help="Only report missing distribution.")
    parser.add_argument("--model", action="append", default=None, help="Model family filter, repeatable.")
    parser.add_argument("--model-version", action="append", default=None, help="Model version filter, repeatable.")
    parser.add_argument("--language", action="append", default=None, help="Language filter (zh/en), repeatable.")
    parser.add_argument("--base", type=Path, default=SCORING_BASE, help="Scoring directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    defaults, models = load_model_settings()
    model_map = {((m.family or m.name).lower(), (m.version or "default").lower()): m for m in models}

    family_filter = {x.lower() for x in args.model} if args.model else None
    version_filter = {x.lower() for x in args.model_version} if args.model_version else None
    language_filter = {x.lower() for x in args.language} if args.language else None

    if not args.base.exists():
        raise SystemExit(f"Missing directory: {args.base}")

    round_idx = 1
    while True:
        jobs = _collect_missing_jobs(
            args.base,
            family_filter=family_filter,
            version_filter=version_filter,
            language_filter=language_filter,
        )
        if args.limit > 0:
            jobs = jobs[: args.limit]

        by_model: Dict[Tuple[str, str], int] = {}
        by_reason: Dict[str, int] = {}
        for j in jobs:
            by_model[(j.family, j.version)] = by_model.get((j.family, j.version), 0) + 1
            by_reason[j.reason] = by_reason.get(j.reason, 0) + 1

        print(f"[Round {round_idx}] real-missing rows: {len(jobs)}")
        for (fam, ver), cnt in sorted(by_model.items()):
            print(f"  - {fam}/{ver}: {cnt}")
        print("  reasons:", ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items())))

        if args.dry_run or not jobs:
            break

        updates: Dict[Path, Dict[int, Dict]] = {}
        failures: List[Tuple[MissingJob, str]] = []
        client_cache: Dict[Tuple[str, str], object] = {}
        temperature = float(defaults.get("temperature", 0.8))
        max_tokens = int(defaults.get("max_output_tokens", 200))

        def run_job(job: MissingJob) -> Tuple[MissingJob, Dict]:
            model = model_map.get((job.family, job.version))
            if model is None:
                raise KeyError(f"Model not found in config: {job.family}/{job.version}")
            cache_key = (job.family, job.version)
            client = client_cache.get(cache_key)
            if client is None:
                client = build_client_for_model(model)
                client_cache[cache_key] = client

            spec = _build_spec_from_record(job.record)
            prompt = build_prompt(spec)
            response = _send_with_retry(client, prompt, temperature=temperature, max_tokens=max_tokens)
            parsed = _parse_score_from_text(response)

            rec = dict(job.record)
            rec["timestamp"] = timestamp()
            rec["score_response"] = str(response).strip()
            if parsed is not None:
                rec["score"] = parsed
                rec.pop("parse_error", None)
                rec.pop("error", None)
            else:
                rec["parse_error"] = "numeric_score_not_found"
            rec["backfill_round"] = round_idx
            return job, rec

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_map = {executor.submit(run_job, job): job for job in jobs}
            with tqdm(total=len(jobs), desc=f"backfill round {round_idx}", unit="row") as pbar:
                for future in as_completed(future_map):
                    job = future_map[future]
                    try:
                        j, rec = future.result()
                        updates.setdefault(j.path, {})[j.line_index] = rec
                    except Exception as exc:  # noqa: BLE001
                        failures.append((job, str(exc)))
                    pbar.update(1)

        patched = _patch_files(updates)
        print(f"[Round {round_idx}] patched rows: {patched}, failures: {len(failures)}")
        if failures:
            out = Path("analysis/resume_backfill_failures.json")
            payload = [
                {
                    "file": str(job.path).replace("\\", "/"),
                    "line": job.line_index + 1,
                    "family": job.family,
                    "version": job.version,
                    "spec_id": job.record.get("spec_id"),
                    "error": err,
                }
                for job, err in failures
            ]
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  failures written: {out}")

        round_idx += 1
        if round_idx > args.max_rounds:
            print(f"Reached max rounds: {args.max_rounds}")
            break


if __name__ == "__main__":
    main()
