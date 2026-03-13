from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from clients import ModelConfig, build_client
from midstream_task import MidstreamRunner, SCENARIOS, SCENARIO_LOOKUP, build_prompt_text, localize_option
from utils import load_model_settings, timestamp


RESULTS_ROOT = Path("data/results_midstream")
ALLOWED_ASSIGNMENT_NAMES = {"小刚", "小婷", "Bob", "Mary"}
NAMES_BY_LANGUAGE = {
    "zh": ("小刚", "小婷"),
    "en": ("Bob", "Mary"),
}


@dataclass
class RowState:
    line_index: int  # 0-based raw line index in file
    record: Dict
    valid: bool
    reasons: List[str]


@dataclass
class BackfillJob:
    model: ModelConfig
    scenario_id: str
    scenario_title: str
    language: str
    combo_key: str
    run_index: int
    option_a_raw: str
    option_b_raw: str
    option_a: str
    option_b: str
    name_order: Tuple[str, str]
    prompt: str
    path: Path
    replace_line_index: Optional[int]
    reason: str
    existing_record: Optional[Dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and loop-backfill strict missing/incomplete midstream rows for all models."
    )
    parser.add_argument("--root", type=Path, default=RESULTS_ROOT, help="Midstream results root directory.")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers.")
    parser.add_argument("--max-rounds", type=int, default=20, help="Maximum loop rounds.")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-round cap (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Only detect/report; do not call API.")
    parser.add_argument("--single-pass", action="store_true", help="Run one round only.")
    parser.add_argument(
        "--strict-rationale-period",
        dest="strict_rationale_period",
        action="store_true",
        default=True,
        help="Require rationale last character to be a period ('.' or '。').",
    )
    parser.add_argument(
        "--no-strict-rationale-period",
        dest="strict_rationale_period",
        action="store_false",
        help="Disable rationale period check.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=0,
        help="Expected runs per combo. 0 = use config defaults (runs_per_name_order * 2).",
    )
    parser.add_argument("--model", action="append", default=None, help="Model family filter. Repeatable.")
    parser.add_argument("--model-version", action="append", default=None, help="Model version filter. Repeatable.")
    parser.add_argument("--language", action="append", default=None, help="Language filter (zh/en). Repeatable.")
    parser.add_argument("--scenario", action="append", default=None, help="Scenario filter (e.g. s1_major).")
    parser.add_argument("--combo", action="append", default=None, help="Combo filter (e.g. combo01). Repeatable.")
    return parser.parse_args()


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


def _filter_match(value: str, allowed: Optional[set[str]]) -> bool:
    if not allowed:
        return True
    return value.lower() in allowed


def _clean_tail(text: str) -> str:
    return text.strip().rstrip('"\'”’）」]） ')


def _rationale_has_period(text: str) -> bool:
    if not text:
        return False
    cleaned = _clean_tail(text)
    return cleaned.endswith(".") or cleaned.endswith("。")


def _validate_record(
    rec: Dict,
    option_a: str,
    option_b: str,
    strict_rationale_period: bool,
) -> List[str]:
    reasons: List[str] = []

    if str(rec.get("error") or "").strip():
        reasons.append("error")
    if str(rec.get("parse_error") or "").strip():
        reasons.append("parse_error")

    response = str(rec.get("response") or "")
    if not response.strip():
        reasons.append("empty_response")

    parsed = rec.get("parsed")
    if not isinstance(parsed, dict):
        reasons.append("parsed_not_dict")
        return reasons

    if not str(parsed.get("xiaogang_profile") or "").strip():
        reasons.append("missing_xiaogang_profile")
    if not str(parsed.get("xiaoting_profile") or "").strip():
        reasons.append("missing_xiaoting_profile")

    assignment = parsed.get("assignment")
    if not isinstance(assignment, dict):
        reasons.append("assignment_not_dict")
    else:
        a_val = assignment.get(option_a)
        b_val = assignment.get(option_b)
        if not str(a_val or "").strip():
            reasons.append("missing_assignment_option_a")
        if not str(b_val or "").strip():
            reasons.append("missing_assignment_option_b")
        if str(a_val or "").strip() and a_val not in ALLOWED_ASSIGNMENT_NAMES:
            reasons.append("invalid_assignment_name_option_a")
        if str(b_val or "").strip() and b_val not in ALLOWED_ASSIGNMENT_NAMES:
            reasons.append("invalid_assignment_name_option_b")

    rationale = str(parsed.get("rationale") or "")
    if not rationale.strip():
        reasons.append("missing_rationale")
    elif strict_rationale_period and not _rationale_has_period(rationale):
        reasons.append("rationale_not_period")

    return reasons


def _name_order_for_run(language: str, run_index: int, expected_runs: int) -> Tuple[str, str]:
    first, second = NAMES_BY_LANGUAGE[language]
    if run_index > (expected_runs // 2):
        return second, first
    return first, second


def _scenario_combo_data(scenario_id: str, combo_key: str, language: str) -> Tuple[str, str, str, str]:
    scenario = SCENARIO_LOOKUP[scenario_id]
    combo_idx = int(combo_key.replace("combo", ""))
    option_a_raw, option_b_raw = scenario.option_pairs[combo_idx - 1]
    option_a = localize_option(option_a_raw, language)
    option_b = localize_option(option_b_raw, language)
    return option_a_raw, option_b_raw, option_a, option_b


def _target_file(root: Path, model: ModelConfig, scenario_id: str, language: str, combo_key: str) -> Path:
    family = (model.family or model.name).lower()
    version = (model.version or "default").lower()
    return root / family / version / scenario_id / language / combo_key / "standard.jsonl"


def _read_row_states(
    path: Path,
    option_a: str,
    option_b: str,
    strict_rationale_period: bool,
) -> Tuple[Dict[int, List[RowState]], Counter]:
    by_run: Dict[int, List[RowState]] = defaultdict(list)
    reason_counter: Counter = Counter()
    if not path.exists():
        return by_run, reason_counter

    with path.open("r", encoding="utf-8") as fh:
        for line_index, line in enumerate(fh):
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except Exception:
                reason_counter["json_decode_fail"] += 1
                continue

            run_index = rec.get("run_index")
            if not isinstance(run_index, int):
                reason_counter["invalid_run_index"] += 1
                continue

            reasons = _validate_record(rec, option_a, option_b, strict_rationale_period)
            valid = len(reasons) == 0
            if not valid:
                reason_counter.update(reasons)
            by_run[run_index].append(RowState(line_index=line_index, record=rec, valid=valid, reasons=reasons))

    return by_run, reason_counter


def collect_backfill_jobs(
    root: Path,
    models: Sequence[ModelConfig],
    scenario_ids: Sequence[str],
    languages: Sequence[str],
    expected_runs: int,
    strict_rationale_period: bool,
    combo_filter: Optional[set[str]],
) -> Tuple[List[BackfillJob], Dict[Tuple[str, str], int], Counter]:
    jobs: List[BackfillJob] = []
    by_model_lang: Dict[Tuple[str, str], int] = defaultdict(int)
    reason_counter: Counter = Counter()

    for model in models:
        for scenario_id in scenario_ids:
            scenario = SCENARIO_LOOKUP[scenario_id]
            for language in languages:
                for combo_idx, _ in enumerate(scenario.option_pairs, start=1):
                    combo_key = f"combo{combo_idx:02d}"
                    if combo_filter and combo_key.lower() not in combo_filter:
                        continue
                    option_a_raw, option_b_raw, option_a, option_b = _scenario_combo_data(
                        scenario_id, combo_key, language
                    )
                    path = _target_file(root, model, scenario_id, language, combo_key)
                    by_run, local_reasons = _read_row_states(path, option_a, option_b, strict_rationale_period)
                    reason_counter.update(local_reasons)

                    for run_index in range(1, expected_runs + 1):
                        states = by_run.get(run_index, [])
                        if any(state.valid for state in states):
                            continue

                        if states:
                            target = states[-1]
                            replace_line_index = target.line_index
                            reason = ",".join(target.reasons) if target.reasons else "invalid_record"
                            existing_record = target.record
                            if target.reasons:
                                reason_counter.update(target.reasons)
                        else:
                            replace_line_index = None
                            reason = "missing_run_index_record"
                            existing_record = None
                            reason_counter["missing_run_index_record"] += 1

                        if isinstance(existing_record, dict):
                            raw_order = existing_record.get("name_order")
                            if isinstance(raw_order, list) and len(raw_order) == 2:
                                name_order = (str(raw_order[0]), str(raw_order[1]))
                            elif isinstance(raw_order, tuple) and len(raw_order) == 2:
                                name_order = (str(raw_order[0]), str(raw_order[1]))
                            else:
                                name_order = _name_order_for_run(language, run_index, expected_runs)
                        else:
                            name_order = _name_order_for_run(language, run_index, expected_runs)

                        prompt = build_prompt_text(scenario, option_a, option_b, name_order, language)
                        job = BackfillJob(
                            model=model,
                            scenario_id=scenario_id,
                            scenario_title=scenario.title_for(language),
                            language=language,
                            combo_key=combo_key,
                            run_index=run_index,
                            option_a_raw=option_a_raw,
                            option_b_raw=option_b_raw,
                            option_a=option_a,
                            option_b=option_b,
                            name_order=name_order,
                            prompt=prompt,
                            path=path,
                            replace_line_index=replace_line_index,
                            reason=reason,
                            existing_record=existing_record,
                        )
                        jobs.append(job)
                        key = ((model.family or model.name).lower(), model.version or "default")
                        by_model_lang[(f"{key[0]}/{key[1]}", language)] += 1

    return jobs, by_model_lang, reason_counter


def _build_record_from_job(job: BackfillJob, response: str, parsed: Optional[Dict], parse_error: Optional[str], round_idx: int) -> Dict:
    if isinstance(job.existing_record, dict):
        rec = dict(job.existing_record)
    else:
        rec = {
            "model": job.model.display_name,
            "model_key": job.model.name,
            "scenario_id": job.scenario_id,
            "scenario_title": job.scenario_title,
            "combo_key": job.combo_key,
            "option_a": job.option_a,
            "option_b": job.option_b,
            "name_order": list(job.name_order),
            "run_index": job.run_index,
            "language": job.language,
            "prompt": job.prompt,
        }
        if job.option_a != job.option_a_raw or job.option_b != job.option_b_raw:
            rec["option_a_raw"] = job.option_a_raw
            rec["option_b_raw"] = job.option_b_raw

    rec["timestamp"] = timestamp()
    rec["model"] = job.model.display_name
    rec["model_key"] = job.model.name
    rec["scenario_id"] = job.scenario_id
    rec["scenario_title"] = job.scenario_title
    rec["combo_key"] = job.combo_key
    rec["option_a"] = job.option_a
    rec["option_b"] = job.option_b
    rec["name_order"] = list(job.name_order)
    rec["run_index"] = job.run_index
    rec["language"] = job.language
    rec["prompt"] = job.prompt
    rec["response"] = response
    rec.pop("error", None)

    if parsed is not None:
        rec["parsed"] = parsed
    else:
        rec.pop("parsed", None)
    if parse_error:
        rec["parse_error"] = parse_error
    else:
        rec.pop("parse_error", None)

    rec["backfill"] = {
        "timestamp": timestamp(),
        "round": round_idx,
        "reason": job.reason,
    }
    return rec


def apply_updates(
    updates: Dict[Path, Dict[str, object]],
) -> int:
    patched = 0
    for path, payload in updates.items():
        replace_map: Dict[int, Dict] = payload.get("replace", {})  # type: ignore[assignment]
        append_rows: List[Dict] = payload.get("append", [])  # type: ignore[assignment]

        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = []

        for idx, rec in replace_map.items():
            if 0 <= idx < len(lines):
                lines[idx] = json.dumps(rec, ensure_ascii=False)
                patched += 1

        for rec in append_rows:
            lines.append(json.dumps(rec, ensure_ascii=False))
            patched += 1

        if lines:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")

    return patched


def run_one_round(
    jobs: Sequence[BackfillJob],
    defaults: Dict,
    round_idx: int,
) -> Tuple[int, List[Tuple[BackfillJob, str]]]:
    temperature = float(defaults.get("temperature", 0.8))
    max_tokens = int(defaults.get("max_output_tokens", 3000))
    max_attempts = int(defaults.get("max_attempts", 5))
    backoff_seconds = float(defaults.get("retry_backoff_seconds", 1.0))
    backoff_max = float(defaults.get("retry_backoff_max", 8.0))
    workers = int(defaults.get("worker_count", 1))

    updates: Dict[Path, Dict[str, object]] = {}
    failures: List[Tuple[BackfillJob, str]] = []
    client_cache: Dict[str, object] = {}

    def _get_client(model: ModelConfig):
        key = model.name.lower()
        client = client_cache.get(key)
        if client is None:
            client = build_client(model)
            client_cache[key] = client
        return client

    def _run_job(job: BackfillJob) -> Tuple[BackfillJob, Dict]:
        client = _get_client(job.model)
        response = send_with_retry(
            client=client,
            prompt_text=job.prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            backoff_max=backoff_max,
        )
        parsed, parse_error = MidstreamRunner._parse_response(response, job.option_a, job.option_b)
        rec = _build_record_from_job(job, response, parsed, parse_error, round_idx)
        return job, rec

    effective_workers = max(1, workers)
    if effective_workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            try:
                j, rec = _run_job(job)
                path_payload = updates.setdefault(j.path, {"replace": {}, "append": []})
                if j.replace_line_index is None:
                    path_payload["append"].append(rec)  # type: ignore[index]
                else:
                    path_payload["replace"][j.replace_line_index] = rec  # type: ignore[index]
            except Exception as exc:  # noqa: BLE001
                failures.append((job, str(exc)))
            if idx % 20 == 0 or idx == len(jobs):
                print(f"Processed {idx}/{len(jobs)}")
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_map = {executor.submit(_run_job, job): job for job in jobs}
            done = 0
            for future in as_completed(future_map):
                done += 1
                job = future_map[future]
                try:
                    j, rec = future.result()
                    path_payload = updates.setdefault(j.path, {"replace": {}, "append": []})
                    if j.replace_line_index is None:
                        path_payload["append"].append(rec)  # type: ignore[index]
                    else:
                        path_payload["replace"][j.replace_line_index] = rec  # type: ignore[index]
                except Exception as exc:  # noqa: BLE001
                    failures.append((job, str(exc)))
                if done % 20 == 0 or done == len(jobs):
                    print(f"Processed {done}/{len(jobs)}")

    patched = apply_updates(updates)
    return patched, failures


def select_models(
    models: Sequence[ModelConfig],
    family_filter: Optional[set[str]],
    version_filter: Optional[set[str]],
) -> List[ModelConfig]:
    selected: List[ModelConfig] = []
    for model in models:
        family = (model.family or model.name).lower()
        version = (model.version or "default").lower()
        if family_filter and family not in family_filter:
            continue
        if version_filter and version not in version_filter:
            continue
        selected.append(model)
    return selected


def main() -> None:
    args = parse_args()
    load_dotenv()
    defaults, models = load_model_settings()

    family_filter = {x.lower() for x in args.model} if args.model else None
    version_filter = {x.lower() for x in args.model_version} if args.model_version else None
    language_filter = {x.lower() for x in args.language} if args.language else None
    scenario_filter = {x.lower() for x in args.scenario} if args.scenario else None
    combo_filter = {x.lower() for x in args.combo} if args.combo else None

    selected_models = select_models(models, family_filter, version_filter)
    if not selected_models:
        raise SystemExit("No models matched filters.")

    if scenario_filter:
        unknown = [sid for sid in scenario_filter if sid not in SCENARIO_LOOKUP]
        if unknown:
            raise SystemExit(f"Unknown scenario id(s): {', '.join(sorted(unknown))}")
        scenario_ids = sorted(scenario_filter)
    else:
        scenario_ids = [s.id for s in SCENARIOS]

    if language_filter:
        unknown_lang = [x for x in language_filter if x not in {"zh", "en"}]
        if unknown_lang:
            raise SystemExit(f"Unsupported language(s): {', '.join(sorted(unknown_lang))}")
        languages = sorted(language_filter)
    else:
        languages = ["zh", "en"]

    expected_runs = args.expected_runs
    if expected_runs <= 0:
        expected_runs = int(defaults.get("runs_per_name_order", 5)) * 2
    defaults["worker_count"] = int(args.workers)

    round_idx = 1
    failure_log: List[Dict] = []
    while True:
        jobs, by_model_lang, reasons = collect_backfill_jobs(
            root=args.root,
            models=selected_models,
            scenario_ids=scenario_ids,
            languages=languages,
            expected_runs=expected_runs,
            strict_rationale_period=args.strict_rationale_period,
            combo_filter=combo_filter,
        )

        if args.limit > 0:
            jobs = jobs[: args.limit]

        print(
            f"[Round {round_idx}] jobs={len(jobs)} | models={len(selected_models)} "
            f"| scenarios={len(scenario_ids)} | languages={len(languages)} | expected_runs={expected_runs}"
        )
        if by_model_lang:
            for (model_lang, lang), count in sorted(by_model_lang.items()):
                print(f"  - {model_lang} | {lang}: {count}")
        if reasons:
            print("  reasons:", ", ".join(f"{k}={v}" for k, v in reasons.most_common(12)))

        if not jobs:
            print("No missing/incomplete rows detected. Backfill finished.")
            break

        if args.dry_run:
            for job in jobs[:80]:
                print(
                    f"{job.path.as_posix()} | run={job.run_index} | scenario={job.scenario_id} "
                    f"| lang={job.language} | reason={job.reason} | replace_line={job.replace_line_index}"
                )
            if len(jobs) > 80:
                print(f"... ({len(jobs) - 80} more)")
            break

        patched, failures = run_one_round(jobs=jobs, defaults=defaults, round_idx=round_idx)
        print(f"[Round {round_idx}] patched={patched}, failures={len(failures)}")

        for job, err in failures:
            failure_log.append(
                {
                    "timestamp": timestamp(),
                    "round": round_idx,
                    "path": job.path.as_posix(),
                    "scenario_id": job.scenario_id,
                    "language": job.language,
                    "combo_key": job.combo_key,
                    "run_index": job.run_index,
                    "model_key": job.model.name,
                    "reason": job.reason,
                    "error": err,
                }
            )

        if args.single_pass:
            print("Stopped after single pass by --single-pass.")
            break

        round_idx += 1
        if round_idx > args.max_rounds:
            print(f"Reached max rounds: {args.max_rounds}")
            break

    if failure_log:
        out = Path("analysis/midstream_backfill_failures.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(failure_log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Failure details written: {out.as_posix()}")


if __name__ == "__main__":
    main()
