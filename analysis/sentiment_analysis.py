from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import shapiro, ttest_ind
from tqdm import tqdm

try:
    from textblob import TextBlob
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'textblob'. Install it first, e.g. `pip install textblob`.") from exc

try:
    from snownlp import SnowNLP
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'snownlp'. Install it first, e.g. `pip install snownlp`.") from exc


MIDSTREAM_ROOT = Path("data/results_midstream")
DOWNSTREAM_ROOT = Path("data/results_downstream")
OUTPUT_DIR = Path("analysis/text_bias/sentiment")
ALL_MODELS = "__ALL_MODELS__"
ALL_DISCIPLINES = "__ALL_DISCIPLINES__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentiment analysis + Shapiro-Wilk + independent t-test for midstream/downstream texts."
    )
    parser.add_argument("--midstream-root", type=Path, default=MIDSTREAM_ROOT)
    parser.add_argument("--downstream-root", type=Path, default=DOWNSTREAM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--task", choices=["midstream", "downstream", "both"], default="both")
    parser.add_argument("--language", choices=["zh", "en", "all"], default="all")
    parser.add_argument("--model", action="append", default=None, help="Model substring filter.")
    parser.add_argument("--shapiro-max-n", type=int, default=5000, help="Max sample size for Shapiro test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Shapiro downsampling.")
    return parser.parse_args()


def _iter_records(root: Path, desc: str) -> Iterable[Tuple[Path, Dict]]:
    files = sorted(root.rglob("standard.jsonl"))
    for path in tqdm(files, desc=desc, unit="file"):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    yield path, rec


def _model_label(rec: Dict, path: Path) -> str:
    model = str(rec.get("model") or "").strip()
    if model:
        return model
    key = str(rec.get("model_key") or "").strip()
    if key:
        return key
    parts = path.parts
    for anchor in ("results_midstream", "results_downstream"):
        if anchor in parts:
            idx = parts.index(anchor)
            if idx + 2 < len(parts):
                return f"{parts[idx+1]}/{parts[idx+2]}"
    return "unknown"


def _extract_midstream_profiles(rec: Dict) -> Tuple[str, str]:
    parsed = rec.get("parsed")
    if isinstance(parsed, dict):
        male = str(parsed.get("xiaogang_profile") or "").strip()
        female = str(parsed.get("xiaoting_profile") or "").strip()
        if male or female:
            return male, female

    response = str(rec.get("response") or "")
    language = str(rec.get("language") or "zh").strip().lower()
    if not response:
        return "", ""

    if language == "en":
        male_labels = r"(?:Bob\s*profile|Bob)"
        female_labels = r"(?:Mary\s*profile|Mary)"
        stop_labels = r"(?:Bob\s*profile|Mary\s*profile|Assignment|Rationale)"
    else:
        male_labels = r"(?:小刚简介|小刚简历|小刚介绍)"
        female_labels = r"(?:小婷简介|小婷简历|小婷介绍)"
        stop_labels = r"(?:小刚简介|小刚简历|小刚介绍|小婷简介|小婷简历|小婷介绍|分配|理由)"

    male_match = re.search(
        rf"{male_labels}\s*[:：]\s*(.+?)(?={stop_labels}\s*[:：]|$)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    female_match = re.search(
        rf"{female_labels}\s*[:：]\s*(.+?)(?={stop_labels}\s*[:：]|$)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    male = male_match.group(1).strip() if male_match else ""
    female = female_match.group(1).strip() if female_match else ""
    return male, female


def _gender_from_downstream(rec: Dict, path: Path) -> Optional[str]:
    person = str(rec.get("person") or "").strip().lower()
    if person == "xiaogang":
        return "male"
    if person == "xiaoting":
        return "female"
    person_name = str(rec.get("person_name") or "").strip().lower()
    if person_name in {"bob", "xiaogang", "小刚"}:
        return "male"
    if person_name in {"mary", "xiaoting", "小婷"}:
        return "female"
    folder = path.parent.name.lower()
    if folder == "xiaogang":
        return "male"
    if folder == "xiaoting":
        return "female"
    return None


def _polarity(text: str, language: str) -> float:
    # TextBlob is used for English; SnowNLP is used for Chinese.
    if (language or "").lower().startswith("zh"):
        # SnowNLP sentiment is in [0, 1]; map to [-1, 1].
        p = float(SnowNLP(text).sentiments)
        return (2.0 * p) - 1.0
    return float(TextBlob(text).sentiment.polarity)


def collect_sentiment_rows(
    task: str,
    root: Path,
    language_filter: Optional[set[str]],
    model_filters: Optional[List[str]],
) -> List[Dict]:
    rows: List[Dict] = []
    for path, rec in _iter_records(root, desc=f"scan {task}"):
        language = str(rec.get("language") or "").strip().lower()
        if language not in {"zh", "en"}:
            continue
        if language_filter and language not in language_filter:
            continue
        model = _model_label(rec, path)
        if model_filters and not any(x in model.lower() for x in model_filters):
            continue

        if task == "midstream":
            discipline = str(rec.get("scenario_id") or "unknown")
            male_text, female_text = _extract_midstream_profiles(rec)
            if male_text:
                rows.append(
                    {
                        "task": task,
                        "language": language,
                        "model": model,
                        "discipline": discipline,
                        "gender": "male",
                        "sentiment": _polarity(male_text, language),
                    }
                )
            if female_text:
                rows.append(
                    {
                        "task": task,
                        "language": language,
                        "model": model,
                        "discipline": discipline,
                        "gender": "female",
                        "sentiment": _polarity(female_text, language),
                    }
                )
            continue

        text = str(rec.get("response") or "").strip()
        if not text:
            continue
        gender = _gender_from_downstream(rec, path)
        if gender is None:
            continue
        discipline = str(rec.get("major") or "unknown")
        rows.append(
            {
                "task": task,
                "language": language,
                "model": model,
                "discipline": discipline,
                "gender": gender,
                "sentiment": _polarity(text, language),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _shapiro(values: List[float], max_n: int, seed: int) -> Tuple[float, float, str]:
    n = len(values)
    if n < 3:
        return float("nan"), float("nan"), "insufficient_n"
    sample = values
    status = "ok"
    if n > max_n:
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        sample = [values[i] for i in idx[:max_n]]
        status = f"downsampled_to_{max_n}"
    stat, p_val = shapiro(sample)
    return float(stat), float(p_val), status


def build_stats(rows: List[Dict], shapiro_max_n: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    # group sentiments for tests
    grouped = defaultdict(list)
    for row in tqdm(rows, desc="group sentiment", unit="row"):
        task = row["task"]
        language = row["language"]
        model = row["model"]
        discipline = row["discipline"]
        gender = row["gender"]
        sentiment = float(row["sentiment"])

        keys = [
            (task, language, model, discipline, gender),
            (task, language, ALL_MODELS, discipline, gender),
            (task, language, model, ALL_DISCIPLINES, gender),
            (task, language, ALL_MODELS, ALL_DISCIPLINES, gender),
        ]
        for key in keys:
            grouped[key].append(sentiment)

    group_rows: List[Dict] = []
    for (task, language, model, discipline, gender), vals in tqdm(
        sorted(grouped.items()), desc="group stats", unit="group"
    ):
        group_rows.append(
            {
                "task": task,
                "language": language,
                "model": model,
                "discipline": discipline,
                "gender": gender,
                "n": len(vals),
                "mean_sentiment": float(mean(vals)),
                "std_sentiment": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            }
        )

    test_rows: List[Dict] = []
    test_keys = sorted({(k[0], k[1], k[2], k[3]) for k in grouped.keys()})
    for task, language, model, discipline in tqdm(test_keys, desc="stat tests", unit="group"):
        male = grouped.get((task, language, model, discipline, "male"), [])
        female = grouped.get((task, language, model, discipline, "female"), [])

        m_w, m_p, m_status = _shapiro(male, max_n=shapiro_max_n, seed=seed)
        f_w, f_p, f_status = _shapiro(female, max_n=shapiro_max_n, seed=seed)
        normal_pass = (
            (len(male) >= 3 and len(female) >= 3)
            and (not np.isnan(m_p))
            and (not np.isnan(f_p))
            and (m_p >= 0.05 and f_p >= 0.05)
        )

        if len(male) >= 2 and len(female) >= 2:
            t_stat, t_p = ttest_ind(male, female, equal_var=False, nan_policy="omit")
            t_stat = float(t_stat)
            t_p = float(t_p)
        else:
            t_stat = float("nan")
            t_p = float("nan")

        male_mean = float(mean(male)) if male else float("nan")
        female_mean = float(mean(female)) if female else float("nan")
        diff = male_mean - female_mean if male and female else float("nan")

        test_rows.append(
            {
                "task": task,
                "language": language,
                "model": model,
                "discipline": discipline,
                "male_n": len(male),
                "female_n": len(female),
                "male_mean": male_mean,
                "female_mean": female_mean,
                "mean_diff_male_minus_female": diff,
                "shapiro_male_W": m_w,
                "shapiro_male_p": m_p,
                "shapiro_male_status": m_status,
                "shapiro_female_W": f_w,
                "shapiro_female_p": f_p,
                "shapiro_female_status": f_status,
                "normality_pass_0_05": normal_pass,
                "t_stat": t_stat,
                "t_pvalue": t_p,
                "t_significant_0_05": (not np.isnan(t_p)) and (t_p < 0.05),
            }
        )

    return group_rows, test_rows


def main() -> None:
    args = parse_args()
    language_filter = None if args.language == "all" else {args.language}
    model_filters = [x.lower() for x in (args.model or [])]

    rows: List[Dict] = []
    if args.task in {"midstream", "both"}:
        rows.extend(
            collect_sentiment_rows(
                task="midstream",
                root=args.midstream_root,
                language_filter=language_filter,
                model_filters=model_filters,
            )
        )
    if args.task in {"downstream", "both"}:
        rows.extend(
            collect_sentiment_rows(
                task="downstream",
                root=args.downstream_root,
                language_filter=language_filter,
                model_filters=model_filters,
            )
        )

    if not rows:
        raise SystemExit("No sentiment rows found after filtering.")

    group_rows, test_rows = build_stats(rows, shapiro_max_n=args.shapiro_max_n, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "sentiment_scores.csv"
    group_path = args.output_dir / "sentiment_group_stats.csv"
    test_path = args.output_dir / "sentiment_tests.csv"

    write_csv(raw_path, rows, ["task", "language", "model", "discipline", "gender", "sentiment"])
    write_csv(
        group_path,
        group_rows,
        ["task", "language", "model", "discipline", "gender", "n", "mean_sentiment", "std_sentiment"],
    )
    write_csv(
        test_path,
        test_rows,
        [
            "task",
            "language",
            "model",
            "discipline",
            "male_n",
            "female_n",
            "male_mean",
            "female_mean",
            "mean_diff_male_minus_female",
            "shapiro_male_W",
            "shapiro_male_p",
            "shapiro_male_status",
            "shapiro_female_W",
            "shapiro_female_p",
            "shapiro_female_status",
            "normality_pass_0_05",
            "t_stat",
            "t_pvalue",
            "t_significant_0_05",
        ],
    )

    print(f"Wrote: {raw_path.as_posix()} (rows={len(rows)})")
    print(f"Wrote: {group_path.as_posix()} (rows={len(group_rows)})")
    print(f"Wrote: {test_path.as_posix()} (rows={len(test_rows)})")


if __name__ == "__main__":
    main()
