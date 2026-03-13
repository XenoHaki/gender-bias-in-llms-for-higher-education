from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

SCORING_BASE = Path("data/results_resume/scoring")
DEFAULT_XLSX_OUTPUT = Path("analysis/resume_scoring_summary.xlsx")
DEFAULT_CSV_OUTPUT = Path("analysis/resume_scoring_summary.csv")
EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_DATA_ROWS = EXCEL_MAX_ROWS - 1  # header row occupies one line

HEADERS = ["序号", "模型", "语言", "姓名", "专业", "绩点", "科研", "实习", "雅思", "简历总分"]


def _parse_score_from_text(text: str) -> Optional[float]:
    content = (text or "").strip()
    if not content:
        return None

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    segments = list(reversed(lines)) if lines else [content]

    def _normalize_numeric(v: float) -> Optional[float]:
        # Resume score should be two-digit (10~100).
        # Exclude accidental IELTS-like tail values such as 6/6.5 picked from rationale.
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

            # Special case requested by user:
            # 4 digits like "4565" => average of "45" and "65" => 55.
            if len(raw) == 4:
                left = int(raw[:2])
                right = int(raw[2:])
                if 0 <= left <= 100 and 0 <= right <= 100:
                    avg = (left + right) / 2.0
                    return _normalize_numeric(avg)

            # Fallback: parse from tail when malformed concatenation appears.
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

        # One-number line with wrappers, e.g. "Score: 85" / "85分".
        one_number = re.fullmatch(r"[^0-9/%-]*([-+]?\d+(?:\.\d+)?)[^0-9/%]*", cleaned)
        if one_number:
            try:
                value = float(one_number.group(1))
            except ValueError:
                return None
            return _normalize_numeric(value)
        return None

    # Prefer the first strict numeric line (many models output score first, then rationale).
    for line in lines:
        v = _parse_single_line_strict(line)
        if v is not None:
            return v

    # Then try from bottom strict numeric lines.
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

            # Special case requested by user:
            # 4 digits like "4565" => average of "45" and "65" => 55.
            if len(raw) == 4:
                left = int(raw[:2])
                right = int(raw[2:])
                if 0 <= left <= 100 and 0 <= right <= 100:
                    avg = (left + right) / 2.0
                    normalized = _normalize_numeric(avg)
                    if normalized is not None:
                        return normalized

            # Fallback: parse from tail when malformed concatenation appears.
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


def _score_value(rec: Dict) -> Tuple[str, bool]:
    # Prefer reparsing score_response to avoid legacy parser artifacts in saved `score`.
    parsed = _parse_score_from_text(str(rec.get("score_response") or ""))
    if parsed is not None:
        score = rec.get("score")
        if isinstance(score, (int, float)) and abs(float(score) - parsed) < 1e-9:
            return f"{parsed:g}", False
        return f"{parsed:g}", True

    # Fallback to existing score only when score_response cannot be parsed.
    score = rec.get("score")
    if isinstance(score, (int, float)):
        score_val = float(score)
        if 10 <= score_val <= 100:
            return f"{score_val:g}", False
    return "", False


def _model_name(rec: Dict, path: Path) -> str:
    model = str(rec.get("model") or "").strip()
    if model:
        return model
    parts = path.parts
    if "scoring" in parts:
        idx = parts.index("scoring")
        if len(parts) > idx + 2:
            return f"{parts[idx + 1]}/{parts[idx + 2]}"
    return ""


def collect_rows(scoring_base: Path) -> Tuple[List[Dict], int]:
    rows: List[Dict] = []
    corrected = 0
    if not scoring_base.exists():
        return rows, corrected

    for path in sorted(scoring_base.rglob("standard.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                score_str, was_corrected = _score_value(rec)
                if was_corrected:
                    corrected += 1

                rows.append(
                    {
                        "模型": _model_name(rec, path),
                        "语言": rec.get("language") or "",
                        "姓名": rec.get("name") or "",
                        "专业": rec.get("major") or "",
                        "绩点": rec.get("gpa") or "",
                        "科研": rec.get("competition") or "",
                        "实习": rec.get("internship") or "",
                        "雅思": rec.get("english") or "",
                        "简历总分": score_str,
                    }
                )

    for i, row in enumerate(rows, start=1):
        row["序号"] = i
    return rows, corrected


def _write_xlsx(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Excel sheet row cap is 1,048,576. For this task we split by language first.
        if len(df) <= EXCEL_MAX_DATA_ROWS:
            df.to_excel(writer, sheet_name="resume_scoring", index=False)
            return

        lang_col = "璇█"
        if lang_col not in df.columns:
            # Fallback: chunk by size when language column is unavailable.
            for idx, start in enumerate(range(0, len(df), EXCEL_MAX_DATA_ROWS), start=1):
                chunk = df.iloc[start : start + EXCEL_MAX_DATA_ROWS]
                chunk.to_excel(writer, sheet_name=f"part_{idx}", index=False)
            return

        grouped = list(df.groupby(lang_col, dropna=False, sort=True))
        wrote_any = False
        part_idx = 1
        for language, sub_df in grouped:
            base_name = f"{str(language or 'unknown')}_scoring"
            # xlsx sheet name max length is 31.
            base_name = base_name[:31]
            if len(sub_df) <= EXCEL_MAX_DATA_ROWS:
                sub_df.to_excel(writer, sheet_name=base_name, index=False)
                wrote_any = True
                continue

            # If a language still exceeds the limit, split into multiple parts.
            for start in range(0, len(sub_df), EXCEL_MAX_DATA_ROWS):
                chunk = sub_df.iloc[start : start + EXCEL_MAX_DATA_ROWS]
                suffix = f"_{part_idx}"
                sheet = f"{base_name[:31 - len(suffix)]}{suffix}"
                chunk.to_excel(writer, sheet_name=sheet, index=False)
                part_idx += 1
                wrote_any = True

        if not wrote_any:
            df.to_excel(writer, sheet_name="resume_scoring", index=False)


def _write_csv(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export resume scoring results to Excel/CSV.")
    parser.add_argument("--input", type=Path, default=SCORING_BASE, help="Input scoring root directory.")
    parser.add_argument(
        "--format",
        choices=["xlsx", "csv"],
        default="xlsx",
        help="Output format. CSV is much faster for very large exports.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (DEFAULT_XLSX_OUTPUT if args.format == "xlsx" else DEFAULT_CSV_OUTPUT)

    rows, corrected = collect_rows(args.input)
    print(f"总记录数: {len(rows)}")
    print(f"检测并修正异常分数条目: {corrected}")

    df = pd.DataFrame(rows, columns=HEADERS)
    if args.format == "xlsx":
        _write_xlsx(df, output)
    else:
        _write_csv(df, output)
    print(f"写入完成: {output}")


if __name__ == "__main__":
    main()
