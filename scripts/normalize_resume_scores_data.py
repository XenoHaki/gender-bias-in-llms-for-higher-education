from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


DEFAULT_BASE = Path("data/results_resume/scoring")


def _parse_score_from_text(text: str) -> Optional[float]:
    content = (text or "").strip()
    if not content:
        return None

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    segments = list(reversed(lines)) if lines else [content]

    def _normalize_numeric(v: float) -> Optional[float]:
        if 10 <= v <= 100:
            return v
        return None

    def _parse_single_line_strict(line: str) -> Optional[float]:
        cleaned = line.strip().rstrip("。.!?;；，,")
        pure_digits = re.fullmatch(r"\d+", cleaned)
        if pure_digits:
            raw = pure_digits.group(0)
            raw_int = int(raw)
            normalized = _normalize_numeric(float(raw_int))
            if normalized is not None:
                return normalized

            # Special-case malformed concatenation, e.g. "4565" -> (45 + 65) / 2.
            if len(raw) == 4:
                left = int(raw[:2])
                right = int(raw[2:])
                if 0 <= left <= 100 and 0 <= right <= 100:
                    return _normalize_numeric((left + right) / 2.0)

            for tail_len in (2, 3):
                if len(raw) >= tail_len:
                    tail = int(raw[-tail_len:])
                    normalized = _normalize_numeric(float(tail))
                    if normalized is not None:
                        return normalized

        pure_number = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned)
        if pure_number:
            try:
                return _normalize_numeric(float(cleaned))
            except ValueError:
                return None

        one_number = re.fullmatch(r"[^0-9/%-]*([-+]?\d+(?:\.\d+)?)[^0-9/%]*", cleaned)
        if one_number:
            try:
                return _normalize_numeric(float(one_number.group(1)))
            except ValueError:
                return None
        return None

    for line in lines:
        value = _parse_single_line_strict(line)
        if value is not None:
            return value

    for line in reversed(lines):
        value = _parse_single_line_strict(line)
        if value is not None:
            return value

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
                    normalized = _normalize_numeric((left + right) / 2.0)
                    if normalized is not None:
                        return normalized

            for tail_len in (2, 3):
                if len(raw) >= tail_len:
                    tail = int(raw[-tail_len:])
                    normalized = _normalize_numeric(float(tail))
                    if normalized is not None:
                        return normalized

        matches = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", segment))
        for match in reversed(matches):
            start = match.start()
            if segment[:start].rstrip().endswith("/"):
                continue
            try:
                value = float(match.group(0))
            except ValueError:
                continue
            normalized = _normalize_numeric(value)
            if normalized is not None:
                return normalized

    return None


def _normalized_score(rec: dict) -> Optional[float]:
    parsed = _parse_score_from_text(str(rec.get("score_response") or ""))
    if parsed is not None:
        return parsed

    score = rec.get("score")
    if isinstance(score, (int, float)):
        score_value = float(score)
        if 10 <= score_value <= 100:
            return score_value
    return None


def process_file(path: Path, dry_run: bool) -> tuple[int, int, int]:
    updated = 0
    removed = 0
    total = 0
    new_lines = []

    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        total += 1
        try:
            rec = json.loads(text)
        except json.JSONDecodeError:
            new_lines.append(text)
            continue

        if not isinstance(rec, dict):
            new_lines.append(text)
            continue

        old_has_score = "score" in rec
        old_score = rec.get("score")
        new_score = _normalized_score(rec)

        changed = False
        if new_score is None:
            if old_has_score:
                rec.pop("score", None)
                removed += 1
                changed = True
        else:
            if not isinstance(old_score, (int, float)) or abs(float(old_score) - new_score) > 1e-12:
                rec["score"] = new_score
                updated += 1
                changed = True

        if dry_run and changed:
            new_lines.append(text)
        else:
            new_lines.append(json.dumps(rec, ensure_ascii=False))

    if not dry_run:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return total, updated, removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply export-time resume score normalization back to JSONL data."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Resume scoring root directory.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without rewriting files.")
    args = parser.parse_args()

    if not args.base.exists():
        raise SystemExit(f"Missing directory: {args.base}")

    files = 0
    rows = 0
    updated = 0
    removed = 0
    changed_files = 0

    for path in sorted(args.base.rglob("standard.jsonl")):
        total, up, rm = process_file(path, dry_run=args.dry_run)
        files += 1
        rows += total
        updated += up
        removed += rm
        if up or rm:
            changed_files += 1
            print(f"{path.as_posix()} | set_score={up} | removed_score={rm}")

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(
        f"[{mode}] files={files}, rows={rows}, changed_files={changed_files}, "
        f"set_score={updated}, removed_score={removed}"
    )


if __name__ == "__main__":
    main()
