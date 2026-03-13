from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


LETTERS_BASE = Path("data/results_downstream")
RESUME_BASE = Path("data/results_resume")
OUTPUT = Path("analysis/downstream_summary.xlsx")
TMP_OUTPUT = Path("analysis/downstream_summary.tmp.xlsx")


MALE_KEYS = {"xiaogang", "bob", "小刚"}
FEMALE_KEYS = {"xiaoting", "mary", "小婷"}


def infer_gender(rec: Dict) -> str:
    raw = str(rec.get("gender") or "").strip()
    if raw:
        return raw

    person = str(rec.get("person") or "").strip().lower()
    person_name = str(rec.get("person_name") or "").strip().lower()
    if person in MALE_KEYS or person_name in MALE_KEYS:
        return "男"
    if person in FEMALE_KEYS or person_name in FEMALE_KEYS:
        return "女"
    return ""


def load_letters() -> List[Dict]:
    rows: List[Dict] = []
    if not LETTERS_BASE.exists():
        return rows

    for path in LETTERS_BASE.rglob("standard.jsonl"):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(
                    {
                        "名字": rec.get("person_name") or rec.get("person") or "",
                        "性别": infer_gender(rec),
                        "模型": rec.get("model") or "",
                        "专业": rec.get("major") or "",
                        "推荐信": rec.get("response") or "",
                    }
                )
    return rows


def load_resume_scores() -> List[Dict]:
    rows: List[Dict] = []
    scoring_dir = RESUME_BASE / "scoring"
    if not scoring_dir.exists():
        return rows

    for path in scoring_dir.rglob("standard.jsonl"):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(
                    {
                        "姓名": rec.get("name") or "",
                        "性别": rec.get("gender") or "",
                        "专业": rec.get("major") or "",
                        "绩点": rec.get("gpa") or "",
                        "科研竞赛": rec.get("competition") or "",
                        "实习经历": rec.get("internship") or "",
                        "英语水平": rec.get("english") or "",
                        "简历分数": rec.get("score_response") or "",
                        "简历内容": rec.get("response") or "",
                    }
                )
    return rows


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    letters = load_letters()
    resumes = load_resume_scores()

    letters_df = pd.DataFrame(letters, columns=["名字", "性别", "模型", "专业", "推荐信"])
    resumes_df = pd.DataFrame(
        resumes,
        columns=["姓名", "性别", "专业", "绩点", "科研竞赛", "实习经历", "英语水平", "简历分数", "简历内容"],
    )

    # Write to temp first, then replace target to avoid half-written broken xlsx.
    with pd.ExcelWriter(TMP_OUTPUT, engine="openpyxl") as writer:
        letters_df.to_excel(writer, sheet_name="letters", index=False)
        resumes_df.to_excel(writer, sheet_name="resume_scores", index=False)

    TMP_OUTPUT.replace(OUTPUT)
    print(f"写入完成: {OUTPUT.as_posix()} | letters={len(letters_df)} | resume_scores={len(resumes_df)}")


if __name__ == "__main__":
    main()

