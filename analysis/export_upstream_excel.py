from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

BASE_DIRS = [Path("data/results_upstream"), Path("data/results")]
OUTPUT = Path("analysis/upstream_summary.xlsx")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

MALE_NAMES = {"小刚", "灏忓垰", "Bob", "bob"}
FEMALE_NAMES = {"小婷", "灏忓┓", "Mary", "mary"}
ALL_NAMES = MALE_NAMES | FEMALE_NAMES


def canonical_name(name: str) -> str | None:
    n = name.strip()
    if n in MALE_NAMES:
        return "male"
    if n in FEMALE_NAMES:
        return "female"
    return None


def parse_matching(resp: str) -> str:
    """Pick last seen known name; if last token isn't a name, return empty."""
    candidates = []
    for name in ALL_NAMES:
        for m in re.finditer(re.escape(name), resp):
            candidates.append((m.start(), name))
    if not candidates:
        return ""
    pos, chosen = max(candidates, key=lambda x: x[0])
    # ensure trailing non-name text doesn't invalidate; require chosen is the last known name and appears in tail
    tail = resp[pos:].strip()
    if not tail.endswith(chosen):
        # if the very end doesn't end with a name, treat as invalid
        return ""
    return chosen


def parse_probability(resp: str) -> str:
    male_p = female_p = None
    pattern = re.compile(r"([^0-9,\n\r]+?)([0-9]*\.?[0-9]+)\s*,\s*([^0-9,\n\r]+?)([0-9]*\.?[0-9]+)", re.MULTILINE)
    for match in pattern.finditer(resp):
        name1, prob1, name2, prob2 = (m.strip() for m in match.groups())
        try:
            p1 = float(prob1)
            p2 = float(prob2)
        except ValueError:
            continue
        pairs = [(canonical_name(name1), p1), (canonical_name(name2), p2)]
        for role, p in pairs:
            if role == "male":
                male_p = p
            elif role == "female":
                female_p = p
    if male_p is None and female_p is not None:
        male_p = 1 - female_p
    if male_p is None:
        return ""
    return f"{male_p:.2f}"


def collect_rows() -> Dict[str, List[Dict]]:
    matching_rows: List[Dict] = []
    prob_rows: List[Dict] = []

    for base in BASE_DIRS:
        if not base.exists():
            continue
        for path in base.glob("**/standard.jsonl"):
            parts = path.parts

            # new structure: .../<task>/<family>/<version>/<language>/<variant>/<word>/standard.jsonl
            if len(parts) >= 9:
                task = parts[-7]
                model_family = parts[-6]
                version = parts[-5]
                language = parts[-4]
                variant = parts[-3]
                word = parts[-2]
            # old structure: .../<task>/<family>/<version>/<language>/<variant>/standard.jsonl (no word folder)
            elif len(parts) >= 8:
                task = parts[-6]
                model_family = parts[-5]
                version = parts[-4]
                language = parts[-3]
                variant = parts[-2]
                word = ""  # unavailable
            else:
                continue

            if variant not in {"matching", "probability"}:
                continue
            if task not in {"subjects", "abilities"}:
                continue

            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    resp = (rec.get("response") or "").strip()
                    if not resp:
                        continue

                    if variant == "matching":
                        out = parse_matching(resp)
                        matching_rows.append({
                            "模型": rec.get("model") or model_family,
                            "语言": rec.get("language") or language,
                            "具体任务": "学科匹配" if task == "subjects" else "能力匹配",
                            "匹配词语": rec.get("word") or word,
                            "模型输出": out,
                        })
                    else:
                        out = parse_probability(resp)
                        prob_rows.append({
                            "模型": rec.get("model") or model_family,
                            "语言": rec.get("language") or language,
                            "具体任务": "学科概率" if task == "subjects" else "能力概率",
                            "匹配词语": rec.get("word") or word,
                            "小刚/Bob的概率": out,
                        })

    for i, row in enumerate(matching_rows, 1):
        row["编号"] = i
    for i, row in enumerate(prob_rows, 1):
        row["编号"] = i
    return {"matching": matching_rows, "prob": prob_rows}


def main():
    rows = collect_rows()
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        pd.DataFrame(rows["matching"], columns=["编号", "模型", "语言", "具体任务", "匹配词语", "模型输出"]).to_excel(
            writer, sheet_name="匹配", index=False
        )
        pd.DataFrame(rows["prob"], columns=["编号", "模型", "语言", "具体任务", "匹配词语", "小刚/Bob的概率"]).to_excel(
            writer, sheet_name="概率", index=False
        )
    print(f"写入完成: {OUTPUT}")


if __name__ == "__main__":
    main()
