from __future__ import annotations

import argparse
import json
import time
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from clients import build_client
from midstream_task import extract_json, extract_kv
from prompts import DatasetBundle, PromptVariant, TaskType, build_prompt
from resume_task import SCORE_PROMPT_TEMPLATE
from utils import load_model_settings


def timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def send_with_retry(client, prompt: str, temperature: float, max_tokens: int, max_attempts: int, backoff: float, backoff_max: float) -> str:
    attempt = 0
    delay = backoff
    while attempt < max_attempts:
        attempt += 1
        try:
            return client.send(prompt, temperature=temperature, max_tokens=max_tokens)
        except Exception:
            if attempt >= max_attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, backoff_max)
    raise RuntimeError("Exceeded retry attempts")


def parse_midstream(response: str, option_a: str, option_b: str):
    data = extract_json(response)
    if data is None:
        data = extract_kv(response, option_a, option_b)
    if data is None:
        return None, "json_decode_error: unable to parse response as JSON or key-value format"

    issues = []
    if "xiaogang_profile" not in data:
        issues.append("missing xiaogang_profile")
    if "xiaoting_profile" not in data:
        issues.append("missing xiaoting_profile")

    assignment = data.get("assignment")
    parsed_assignment = {}
    if isinstance(assignment, dict):
        parsed_assignment[option_a] = assignment.get(option_a)
        parsed_assignment[option_b] = assignment.get(option_b)
        if option_a not in assignment or option_b not in assignment:
            issues.append("assignment missing option key")
    else:
        issues.append("assignment not found or not an object")

    parsed = {
        "xiaogang_profile": data.get("xiaogang_profile"),
        "xiaoting_profile": data.get("xiaoting_profile"),
        "assignment": parsed_assignment,
        "rationale": data.get("rationale"),
    }
    error_text = "; ".join(issues) if issues else None
    return parsed, error_text


def is_numeric_score(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def is_retry_resolved(rec: dict) -> bool:
    has_error = bool(rec.get("error") or rec.get("parse_error"))
    score_ok = True
    if "score_response" in rec:
        score_ok = is_numeric_score(str(rec.get("score_response", "")).strip())
    has_output = bool(rec.get("response") or rec.get("score_response"))
    return (not has_error) and score_ok and has_output


def choose_best_retry(records: list[dict]) -> dict:
    for rec in reversed(records):
        if is_retry_resolved(rec):
            return rec
    return records[-1]


def dedupe_retries_in_file(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        rows = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rows.append(rec)

    retry_groups = defaultdict(list)
    passthrough = []
    for rec in rows:
        retry_of_path = rec.get("retry_of_path")
        retry_of_line = rec.get("retry_of_line")
        if retry_of_path and retry_of_line:
            retry_groups[(retry_of_path, int(retry_of_line))].append(rec)
        else:
            passthrough.append(rec)

    deduped = list(passthrough)
    removed = 0
    for key, group in retry_groups.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            removed += len(group) - 1
            deduped.append(choose_best_retry(group))

    if removed:
        with path.open("w", encoding="utf-8") as fh:
            for rec in deduped:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return removed


def dedupe_retries(root: Path) -> int:
    removed_total = 0
    for path in root.rglob("*.jsonl"):
        removed_total += dedupe_retries_in_file(path)
    return removed_total


def collect_candidates_from_data(root: Path) -> list[dict]:
    jsonl_files = list(root.rglob("*.jsonl"))
    resolved = set()
    candidates = []
    seen = set()

    for path in jsonl_files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                retry_of_path = rec.get("retry_of_path")
                retry_of_line = rec.get("retry_of_line")
                if retry_of_path and retry_of_line:
                    if is_retry_resolved(rec):
                        resolved.add((retry_of_path, int(retry_of_line)))

    for path in jsonl_files:
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue

                key = (str(path), line_no)
                if key in resolved:
                    continue

                if rec.get("error"):
                    item = (
                        str(path),
                        line_no,
                        "error",
                    )
                    if item not in seen:
                        candidates.append(
                            {
                                "path": str(path),
                                "line": line_no,
                                "category": "error",
                                "reason": str(rec.get("error"))[:200],
                            }
                        )
                        seen.add(item)
                if rec.get("parse_error"):
                    item = (
                        str(path),
                        line_no,
                        "parse_error",
                    )
                    if item not in seen:
                        candidates.append(
                            {
                                "path": str(path),
                                "line": line_no,
                                "category": "parse_error",
                                "reason": str(rec.get("parse_error"))[:200],
                            }
                        )
                        seen.add(item)
                if "score_response" in rec:
                    val = str(rec.get("score_response", "")).strip()
                    if val and (not is_numeric_score(val)):
                        item = (
                            str(path),
                            line_no,
                            "score_format",
                        )
                        if item not in seen:
                            candidates.append(
                                {
                                    "path": str(path),
                                    "line": line_no,
                                    "category": "score_format",
                                    "reason": val[:200],
                                }
                            )
                            seen.add(item)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry failed/invalid rows and append to original JSONL files.")
    parser.add_argument(
        "--candidates",
        default="",
        help="Optional path to retry candidate list; if omitted, rescan data/ for remaining issues.",
    )
    parser.add_argument(
        "--models",
        default="deepseek,qwen,kimi,ernie",
        help="Comma-separated model families to retry (e.g. deepseek,qwen).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of retries to run (0 = no limit).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent retries to run.",
    )
    parser.add_argument(
        "--retry-on-error",
        type=int,
        default=1,
        help="Immediate retry count when response still has parse/format errors.",
    )
    parser.add_argument(
        "--dedupe-first",
        action="store_true",
        help="Remove duplicate retry records before running.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.dedupe_first:
        removed = dedupe_retries(Path("data"))
        if removed:
            print(f"Removed {removed} duplicate retry records.")

    if args.candidates:
        candidates_path = Path(args.candidates)
        if not candidates_path.exists():
            raise SystemExit(f"Missing candidates file: {candidates_path}")
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    else:
        candidates = collect_candidates_from_data(Path("data"))
    allowed_families = {name.strip().lower() for name in args.models.split(",") if name.strip()}

    defaults, models = load_model_settings()
    model_by_key = {m.name.lower(): m for m in models}
    model_by_family_ver = {(m.family or m.name).lower() + "|" + (m.version or "default").lower(): m for m in models}

    max_attempts = int(defaults.get("max_attempts", 3))
    backoff = float(defaults.get("retry_backoff_seconds", 1.0))
    backoff_max = float(defaults.get("retry_backoff_max", 8.0))

    counts = {
        "total": 0,
        "skipped_not_allowed": 0,
        "skipped_no_model": 0,
        "skipped_missing_data": 0,
        "skipped_limit": 0,
        "failed": 0,
        "success": 0,
    }
    failures = []
    client_cache = {}
    count_lock = threading.Lock()
    file_lock_map = {}
    file_lock_guard = threading.Lock()

    def get_model_for_record(rec: dict, path: Path):
        key = (rec.get("model_key") or "").lower()
        if key and key in model_by_key:
            return model_by_key[key]
        fam = (rec.get("model_family") or "").lower()
        ver = (rec.get("model_version") or "").lower()
        if fam and ver:
            key2 = f"{fam}|{ver}"
            if key2 in model_by_family_ver:
                return model_by_family_ver[key2]
        parts = path.parts
        if "data" in parts:
            idx = parts.index("data")
            if len(parts) > idx + 3:
                fam2 = parts[idx + 2].lower()
                ver2 = parts[idx + 3].lower()
                key3 = f"{fam2}|{ver2}"
                if key3 in model_by_family_ver:
                    return model_by_family_ver[key3]
        return None

    def get_client(model):
        key = model.name.lower()
        if key not in client_cache:
            client_cache[key] = build_client(model)
        return client_cache[key]

    def get_file_lock(path: Path) -> threading.Lock:
        key = str(path)
        with file_lock_guard:
            if key not in file_lock_map:
                file_lock_map[key] = threading.Lock()
            return file_lock_map[key]

    def process_item(item: dict) -> dict:
        path = Path(item["path"])
        line_no = int(item["line"])
        category = item.get("category")

        if not path.exists():
            return {"status": "skipped_missing_data"}

        rec = None
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                if i == line_no:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        rec = None
                    break

        if not isinstance(rec, dict):
            return {"status": "skipped_missing_data"}

        model = get_model_for_record(rec, path)
        if not model:
            return {"status": "skipped_no_model"}

        family = (model.family or model.name).lower()
        if family not in allowed_families:
            return {"status": "skipped_not_allowed"}

        with count_lock:
            if args.limit and counts["total"] >= args.limit:
                return {"status": "skipped_limit"}
            counts["total"] += 1

        client = get_client(model)
        path_str = str(path).replace("\\", "/")
        attempts_left = max(1, int(args.retry_on_error) + 1)

        while attempts_left > 0:
            attempts_left -= 1
            try:
                payload = dict(rec)
                payload["retry_of_path"] = str(path)
                payload["retry_of_line"] = line_no
                payload["retry_reason"] = category
                payload["retry_timestamp"] = timestamp()

                if "/results_midstream/" in path_str:
                    prompt = rec.get("prompt")
                    if not prompt:
                        return {"status": "skipped_missing_data"}
                    response = send_with_retry(
                        client,
                        prompt,
                        temperature=float(defaults.get("temperature", 0.8)),
                        max_tokens=int(defaults.get("max_output_tokens", 800)),
                        max_attempts=max_attempts,
                        backoff=backoff,
                        backoff_max=backoff_max,
                    )
                    payload["response"] = response
                    option_a = rec.get("option_a")
                    option_b = rec.get("option_b")
                    if option_a and option_b:
                        parsed, parse_error = parse_midstream(response, option_a, option_b)
                        if parsed is not None:
                            payload["parsed"] = parsed
                        if parse_error:
                            payload["parse_error"] = parse_error
                    if payload.get("parse_error") and attempts_left > 0:
                        tqdm.write(f"parse_error retry: {path}:{line_no} ({model.name}) -> {payload.get('parse_error')}")
                        continue

                elif "/results_downstream/" in path_str:
                    prompt = rec.get("prompt")
                    if not prompt:
                        return {"status": "skipped_missing_data"}
                    response = send_with_retry(
                        client,
                        prompt,
                        temperature=float(defaults.get("temperature", 0.8)),
                        max_tokens=int(defaults.get("max_output_tokens", 1200)),
                        max_attempts=max_attempts,
                        backoff=backoff,
                        backoff_max=backoff_max,
                    )
                    payload["response"] = response

                elif "/results_resume/generation/" in path_str:
                    prompt = rec.get("prompt")
                    if not prompt:
                        return {"status": "skipped_missing_data"}
                    response = send_with_retry(
                        client,
                        prompt,
                        temperature=float(defaults.get("temperature", 0.8)),
                        max_tokens=int(defaults.get("max_output_tokens", 1200)),
                        max_attempts=max_attempts,
                        backoff=backoff,
                        backoff_max=backoff_max,
                    )
                    payload["response"] = response

                elif "/results_resume/scoring/" in path_str:
                    resume = rec.get("response")
                    if not resume:
                        return {"status": "skipped_missing_data"}
                    prompt = SCORE_PROMPT_TEMPLATE.format(resume=resume)
                    response = send_with_retry(
                        client,
                        prompt,
                        temperature=float(defaults.get("temperature", 0.8)),
                        max_tokens=int(defaults.get("max_output_tokens", 200)),
                        max_attempts=max_attempts,
                        backoff=backoff,
                        backoff_max=backoff_max,
                    )
                    payload["score_response"] = str(response).strip()
                    if not is_numeric_score(payload["score_response"]) and attempts_left > 0:
                        tqdm.write(f"score_format retry: {path}:{line_no} ({model.name}) -> {payload['score_response'][:80]}")
                        continue

                else:
                    task = rec.get("task")
                    variant = rec.get("variant")
                    language = rec.get("language")
                    words = rec.get("words")
                    name_order = rec.get("name_order")
                    if not (task and variant and language and words and name_order):
                        return {"status": "skipped_missing_data"}
                    if isinstance(name_order, list):
                        name_order = tuple(name_order)
                    bundle = DatasetBundle(
                        task=TaskType(task),
                        variant=PromptVariant(variant),
                        language=language,
                        words=words,
                    )
                    prompt = build_prompt(bundle, name_order)
                    response = send_with_retry(
                        client,
                        prompt,
                        temperature=float(defaults.get("temperature", 0.8)),
                        max_tokens=int(defaults.get("max_output_tokens", 800)),
                        max_attempts=max_attempts,
                        backoff=backoff,
                        backoff_max=backoff_max,
                    )
                    payload["response"] = response

                lock = get_file_lock(path)
                with lock:
                    with path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                return {"status": "success"}
            except Exception as exc:
                if attempts_left > 0:
                    tqdm.write(f"retry error: {path}:{line_no} ({model.name}) -> {exc}")
                    continue
                return {
                    "status": "failed",
                    "path": str(path),
                    "line": line_no,
                    "model": model.name if model else "",
                    "error": str(exc),
                }

    progress = tqdm(total=len(candidates), desc="retry", unit="item")
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(process_item, item) for item in candidates]
            for future in as_completed(futures):
                result = future.result()
                status = result.get("status")
                if status == "success":
                    counts["success"] += 1
                elif status == "failed":
                    counts["failed"] += 1
                    error_msg = (
                        f"retry failed: {result.get('path')}:{result.get('line')} "
                        f"({result.get('model')}) -> {result.get('error')}"
                    )
                    tqdm.write(error_msg)
                    failures.append(result)
                elif status == "skipped_missing_data":
                    counts["skipped_missing_data"] += 1
                elif status == "skipped_no_model":
                    counts["skipped_no_model"] += 1
                elif status == "skipped_not_allowed":
                    counts["skipped_not_allowed"] += 1
                elif status == "skipped_limit":
                    counts["skipped_limit"] += 1

                progress.update(1)
    finally:
        progress.close()

    if failures:
        report_path = Path("notes/retry_failures.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote failure report: {report_path}")

    print("Done")
    print(counts)


if __name__ == "__main__":
    main()
