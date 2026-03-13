from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_ROOT = Path("data/results_midstream/grok")
ZH_MALE = "\u5c0f\u521a"
ZH_FEMALE = "\u5c0f\u5a77"

MALE_ALIASES = (ZH_MALE, "Bob", "Xiaogang")
FEMALE_ALIASES = (ZH_FEMALE, "Mary", "Xiaoting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix Grok midstream parsed.assignment fields that contain __SEMI__ artifacts."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root directory to scan.")
    parser.add_argument("--dry-run", action="store_true", help="Only report potential fixes.")
    return parser.parse_args()


def _canonical_name(raw: str, language: str) -> Optional[str]:
    text = (raw or "").strip()
    if not text:
        return None
    if text in MALE_ALIASES:
        return "Bob" if language == "en" else ZH_MALE
    if text in FEMALE_ALIASES:
        return "Mary" if language == "en" else ZH_FEMALE

    male_hits = [m.start() for m in re.finditer(r"(?:\u5c0f\u521a|Bob|Xiaogang)", text, flags=re.IGNORECASE)]
    female_hits = [m.start() for m in re.finditer(r"(?:\u5c0f\u5a77|Mary|Xiaoting)", text, flags=re.IGNORECASE)]
    last_male = max(male_hits) if male_hits else -1
    last_female = max(female_hits) if female_hits else -1

    if last_male < 0 and last_female < 0:
        return None
    if last_male > last_female:
        return "Bob" if language == "en" else ZH_MALE
    return "Mary" if language == "en" else ZH_FEMALE


def _assignment_looks_broken(parsed: Dict, option_a: str, option_b: str) -> bool:
    assignment = parsed.get("assignment")
    if not isinstance(assignment, dict):
        return True

    a_val = str(assignment.get(option_a) or "")
    b_val = str(assignment.get(option_b) or "")
    combined = f"{a_val} {b_val}"

    if "__SEMI__" in combined:
        return True
    if not a_val.strip() or not b_val.strip():
        return True
    return False


def _extract_assignment_from_text(text: str, option_a: str, option_b: str, language: str) -> Optional[Dict[str, str]]:
    normalized = (text or "").replace("__SEMI__", ";").replace("；", ";").replace("\n", " ")
    if not normalized.strip():
        return None

    found: Dict[str, str] = {}
    for option in (option_a, option_b):
        pattern = re.compile(rf"{re.escape(option)}\s*[=:：]\s*([^;,\n]+)")
        matches = list(pattern.finditer(normalized))
        if not matches:
            continue
        value = matches[-1].group(1).strip()
        name = _canonical_name(value, language)
        if name:
            found[option] = name

    if option_a in found and option_b in found:
        return {option_a: found[option_a], option_b: found[option_b]}

    parts = [part.strip() for part in re.split(r"[;,\n]", normalized) if part.strip()]
    names = [_canonical_name(part, language) for part in parts]
    names = [name for name in names if name]

    if option_a not in found and names:
        found[option_a] = names[0]
    if option_b not in found and len(names) > 1:
        found[option_b] = names[1]

    if option_a in found and option_b in found:
        return {option_a: found[option_a], option_b: found[option_b]}
    return None


def _repair_assignment(rec: Dict) -> Optional[Dict[str, str]]:
    option_a = str(rec.get("option_a") or "").strip()
    option_b = str(rec.get("option_b") or "").strip()
    language = str(rec.get("language") or "zh").strip()
    if not option_a or not option_b:
        return None

    parsed = rec.get("parsed") if isinstance(rec.get("parsed"), dict) else {}
    assignment = parsed.get("assignment") if isinstance(parsed.get("assignment"), dict) else {}

    existing_blob = f"{option_a}={assignment.get(option_a, '')}; {option_b}={assignment.get(option_b, '')}"
    response = str(rec.get("response") or "")

    for candidate in (existing_blob, response):
        fixed = _extract_assignment_from_text(candidate, option_a, option_b, language)
        if fixed:
            return fixed
    return None


def process_file(path: Path, dry_run: bool) -> Tuple[int, int]:
    updated = 0
    scanned = 0
    lines = path.read_text(encoding="utf-8").splitlines()

    for idx, raw in enumerate(lines):
        text = raw.strip()
        if not text:
            continue
        scanned += 1

        try:
            rec = json.loads(text)
        except json.JSONDecodeError:
            continue

        option_a = str(rec.get("option_a") or "").strip()
        option_b = str(rec.get("option_b") or "").strip()
        if not option_a or not option_b:
            continue

        parsed_old = rec.get("parsed")
        if not isinstance(parsed_old, dict):
            parsed_old = {}
        if not _assignment_looks_broken(parsed_old, option_a, option_b):
            continue

        fixed_assignment = _repair_assignment(rec)
        if not fixed_assignment:
            continue

        rec["parsed"] = dict(parsed_old)
        rec["parsed"]["assignment"] = fixed_assignment

        if not dry_run:
            lines[idx] = json.dumps(rec, ensure_ascii=False)
        updated += 1

    if updated and not dry_run:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return scanned, updated


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise SystemExit(f"Missing directory: {args.root}")

    total_files = 0
    total_scanned = 0
    total_updated = 0

    for path in sorted(args.root.rglob("standard.jsonl")):
        scanned, updated = process_file(path, dry_run=args.dry_run)
        total_files += 1
        total_scanned += scanned
        total_updated += updated
        if updated:
            print(f"{path.as_posix()} | fixed={updated}")

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(
        f"[{mode}] files={total_files}, scanned_rows={total_scanned}, assignment_fixed_rows={total_updated}"
    )


if __name__ == "__main__":
    main()
