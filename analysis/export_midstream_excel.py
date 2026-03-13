from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from midstream_task import extract_json, extract_kv, extract_loose

BASE = Path("data/results_midstream")
OUTPUT = Path("analysis/midstream_summary.xlsx")

HEADERS = ["序号", "模型", "语言", "具体情境", "小刚简介", "小婷简介", "小刚", "小婷", "匹配理由"]


def canonical_name(name: str) -> Optional[str]:
    raw = (name or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    compact = re.sub(r"\s+", "", raw)
    compact_lower = compact.lower()

    if "小刚" in compact or "灏忓垰" in compact or "bob" in compact_lower or "xiaogang" in compact_lower:
        return "male"
    if "小婷" in compact or "灏忓┓" in compact or "mary" in compact_lower or "xiaoting" in compact_lower:
        return "female"
    return None


def parse_from_response(response: str, option_a: str, option_b: str) -> Optional[Dict]:
    text = (response or "").strip()
    if not text:
        return None

    data = extract_json(text)
    if data is None:
        data = extract_kv(text, option_a, option_b)
    if data is None:
        data = extract_loose(text, option_a, option_b)
    if not isinstance(data, dict):
        return None
    return data


def _pick_profile(parsed: Dict, fallback: Dict, key: str) -> str:
    value = parsed.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = fallback.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _pick_rationale(parsed: Dict, fallback: Dict, parse_error: str) -> str:
    value = parsed.get("rationale")
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = fallback.get("rationale")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return parse_error or ""


def _pick_assignment(parsed: Dict, fallback: Dict) -> Dict:
    assign = parsed.get("assignment")
    if isinstance(assign, dict) and assign:
        return assign
    fallback_assign = fallback.get("assignment")
    if isinstance(fallback_assign, dict):
        return fallback_assign
    return {}


def load_rows() -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(BASE.rglob("standard.jsonl")):
        candidate_lang = path.parent.parent.name if path.parent.parent else ""
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                option_a = str(rec.get("option_a") or "")
                option_b = str(rec.get("option_b") or "")
                response = str(rec.get("response") or "")
                parse_error = str(rec.get("parse_error") or "")

                parsed_raw = rec.get("parsed")
                parsed = parsed_raw if isinstance(parsed_raw, dict) else {}
                fallback = parse_from_response(response, option_a, option_b) or {}

                assignment = _pick_assignment(parsed, fallback)
                xg_profile = _pick_profile(parsed, fallback, "xiaogang_profile")
                xt_profile = _pick_profile(parsed, fallback, "xiaoting_profile")
                rationale = _pick_rationale(parsed, fallback, parse_error)

                male_word = ""
                female_word = ""
                for word, person in assignment.items():
                    if not word:
                        continue
                    canon = canonical_name(str(person))
                    if canon == "male":
                        male_word = str(word)
                    elif canon == "female":
                        female_word = str(word)

                language = rec.get("language")
                if not language:
                    language = candidate_lang if candidate_lang in {"zh", "en"} else "zh"

                rows.append(
                    {
                        "模型": rec.get("model") or "",
                        "语言": language,
                        "具体情境": f"{rec.get('scenario_id')}|{rec.get('scenario_title')}|{rec.get('combo_key')}",
                        "小刚简介": xg_profile or response,
                        "小婷简介": xt_profile,
                        "小刚": male_word,
                        "小婷": female_word,
                        "匹配理由": rationale,
                    }
                )
    for idx, row in enumerate(rows, start=1):
        row["序号"] = idx
    return rows


def main() -> None:
    rows = load_rows()
    df = pd.DataFrame(rows, columns=HEADERS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT, index=False)
    print(f"写入完成: {OUTPUT.as_posix()} (rows={len(rows)})")


if __name__ == "__main__":
    main()

