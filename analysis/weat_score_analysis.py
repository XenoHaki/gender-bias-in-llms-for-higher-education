from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


DEFAULT_LOGOR = Path("analysis/text_bias/logor_results.csv")
DEFAULT_ATTR = Path("analysis/weat_attribute_sets.json")
DEFAULT_GLOVE = Path("data/embeddings/glove.840B.300d.txt")
DEFAULT_FASTTEXT_EN = Path("data/embeddings/cc.en.300.vec")
DEFAULT_FASTTEXT_ZH = Path("data/embeddings/cc.zh.300.vec")
DEFAULT_OUTDIR = Path("analysis/text_bias/weat")

ALL_MODELS = "__ALL_MODELS__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WEAT-style score analysis with GloVe embeddings.")
    parser.add_argument("--logor-file", type=Path, default=DEFAULT_LOGOR, help="Input log-OR CSV from step 1.")
    parser.add_argument(
        "--attributes-file",
        type=Path,
        default=DEFAULT_ATTR,
        help="Attribute sets json file. Edit this to customize A/B attributes.",
    )
    parser.add_argument(
        "--glove-file",
        type=Path,
        default=DEFAULT_GLOVE,
        help="Path to GloVe vectors (recommended: glove.840B.300d.txt).",
    )
    parser.add_argument(
        "--fasttext-en-file",
        type=Path,
        default=DEFAULT_FASTTEXT_EN,
        help="Path to English fastText .vec file.",
    )
    parser.add_argument(
        "--fasttext-zh-file",
        type=Path,
        default=DEFAULT_FASTTEXT_ZH,
        help="Path to Chinese fastText .vec file.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "glove", "fasttext"],
        default="auto",
        help="Embedding backend. 'auto' uses GloVe for English and fastText for Chinese.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--task", choices=["midstream", "downstream", "both"], default="both")
    parser.add_argument("--language", choices=["en", "zh", "all"], default="en")
    parser.add_argument("--model", default=ALL_MODELS, help="Model filter. Use __ALL_MODELS__ for merged data.")
    parser.add_argument("--top-k", type=int, default=200, help="Top-K male/female target adjectives from log-OR.")
    parser.add_argument("--min-abs-logor", type=float, default=0.1, help="Minimum absolute log-OR.")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum count_m + count_f for target words.")
    parser.add_argument("--target-lower", action="store_true", default=True, help="Lowercase target words.")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding for log-OR input.")
    return parser.parse_args()


def _normalize_word(word: str, to_lower: bool) -> str:
    w = (word or "").strip()
    return w.lower() if to_lower else w


def _count_csv_rows(path: Path, encoding: str) -> int:
    with path.open("r", encoding=encoding, newline="") as fh:
        # minus header
        return max(sum(1 for _ in fh) - 1, 0)


def load_logor_targets(
    path: Path,
    task_filter: str,
    language_filter: str,
    model_filter: str,
    top_k: int,
    min_abs_logor: float,
    min_count: int,
    to_lower: bool,
    encoding: str,
) -> Tuple[List[str], List[str]]:
    male_rows = []
    female_rows = []
    total_rows = _count_csv_rows(path, encoding=encoding)
    with path.open("r", encoding=encoding, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in tqdm(reader, total=total_rows, desc="load log-or", unit="row"):
            task = (row.get("task") or "").strip()
            language = (row.get("language") or "").strip()
            model = (row.get("model") or "").strip()
            token = _normalize_word(row.get("token") or "", to_lower)
            if not token:
                continue

            if task_filter != "both" and task != task_filter:
                continue
            if language_filter != "all" and language != language_filter:
                continue
            if model_filter != ALL_MODELS and model != model_filter:
                continue
            if model_filter == ALL_MODELS and model != ALL_MODELS:
                continue

            try:
                log_or = float(row.get("log_or") or "nan")
                c_m = int(float(row.get("count_m") or "0"))
                c_f = int(float(row.get("count_f") or "0"))
            except ValueError:
                continue
            if not math.isfinite(log_or):
                continue
            if c_m + c_f < min_count:
                continue
            if abs(log_or) < min_abs_logor:
                continue

            entry = (token, log_or, c_m, c_f)
            if log_or > 0:
                male_rows.append(entry)
            elif log_or < 0:
                female_rows.append(entry)

    male_rows.sort(key=lambda x: x[1], reverse=True)
    female_rows.sort(key=lambda x: x[1])  # more negative = stronger female-biased

    male_targets = [x[0] for x in male_rows[:top_k]]
    female_targets = [x[0] for x in female_rows[:top_k]]
    return male_targets, female_targets


def load_attribute_sets(path: Path, language: str) -> Dict[str, Dict[str, List[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, List[str]]] = {}
    for key, block in payload.items():
        if language in {"en", "zh"} and isinstance(block.get(language), dict):
            source = block[language]
        else:
            source = block
        a_words = [str(x).strip().lower() for x in source.get("A", []) if str(x).strip()]
        b_words = [str(x).strip().lower() for x in source.get("B", []) if str(x).strip()]
        out[key] = {
            "description": str(block.get("description") or ""),
            "A": a_words,
            "B": b_words,
        }
    return out


def select_embedding_file(args: argparse.Namespace) -> Tuple[str, Path]:
    if args.embedding_backend == "glove":
        return "glove", args.glove_file
    if args.embedding_backend == "fasttext":
        if args.language == "zh":
            return "fasttext", args.fasttext_zh_file
        if args.language == "en":
            return "fasttext", args.fasttext_en_file
        raise SystemExit("fastText backend requires --language en or --language zh.")
    if args.language == "zh":
        return "fasttext", args.fasttext_zh_file
    if args.language == "en":
        return "glove", args.glove_file
    raise SystemExit("--language all is not supported with language-specific WEAT attributes. Use en or zh.")


def collect_needed_vocab(
    male_targets: List[str],
    female_targets: List[str],
    attr_sets: Dict[str, Dict[str, List[str]]],
) -> set[str]:
    vocab = set(male_targets) | set(female_targets)
    for block in attr_sets.values():
        vocab.update(block["A"])
        vocab.update(block["B"])
    return {x for x in vocab if x}


def load_glove_subset(glove_path: Path, needed_words: set[str]) -> Dict[str, np.ndarray]:
    if not glove_path.exists():
        raise SystemExit(f"Missing glove file: {glove_path}")
    vectors: Dict[str, np.ndarray] = {}

    total_bytes = glove_path.stat().st_size
    with glove_path.open("r", encoding="utf-8", errors="ignore") as fh, tqdm(
        total=total_bytes, unit="B", unit_scale=True, desc="load glove subset"
    ) as pbar:
        for line in fh:
            pbar.update(len(line.encode("utf-8", errors="ignore")))
            if len(vectors) == len(needed_words):
                break
            parts = line.rstrip().split(" ")
            if len(parts) < 11:
                continue
            word = parts[0]
            if word not in needed_words:
                continue
            try:
                vec = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vectors[word] = vec / norm
    return vectors


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v))


def association_score(w: np.ndarray, a_vecs: List[np.ndarray], b_vecs: List[np.ndarray]) -> float:
    if not a_vecs or not b_vecs:
        return float("nan")
    a_mean = float(np.mean([cosine(w, a) for a in a_vecs]))
    b_mean = float(np.mean([cosine(w, b) for b in b_vecs]))
    return a_mean - b_mean


def safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    attr_sets = load_attribute_sets(args.attributes_file, args.language)

    male_targets, female_targets = load_logor_targets(
        path=args.logor_file,
        task_filter=args.task,
        language_filter=args.language,
        model_filter=args.model,
        top_k=args.top_k,
        min_abs_logor=args.min_abs_logor,
        min_count=args.min_count,
        to_lower=args.target_lower,
        encoding=args.encoding,
    )

    if not male_targets or not female_targets:
        raise SystemExit(
            "No target words selected from log-OR results. Relax filters (task/language/model/top-k/min-count)."
        )

    needed = collect_needed_vocab(male_targets, female_targets, attr_sets)
    backend_name, embedding_file = select_embedding_file(args)
    vectors = load_glove_subset(embedding_file, needed_words=needed)

    missing_targets = [w for w in male_targets + female_targets if w not in vectors]
    if missing_targets:
        print(f"[warn] Missing target vectors: {len(missing_targets)}")

    word_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for set_name, block in tqdm(attr_sets.items(), desc="compute weat", unit="set"):
        a_words = [w for w in block["A"] if w in vectors]
        b_words = [w for w in block["B"] if w in vectors]
        a_vecs = [vectors[w] for w in a_words]
        b_vecs = [vectors[w] for w in b_words]

        male_scores: List[float] = []
        female_scores: List[float] = []

        for token in tqdm(male_targets, desc=f"{set_name} male", unit="tok", leave=False):
            vec = vectors.get(token)
            if vec is None:
                continue
            score = association_score(vec, a_vecs, b_vecs)
            if not math.isfinite(score):
                continue
            male_scores.append(score)
            word_rows.append(
                {
                    "attribute_set": set_name,
                    "target_group": "male",
                    "token": token,
                    "score_s_w_A_B": score,
                }
            )

        for token in tqdm(female_targets, desc=f"{set_name} female", unit="tok", leave=False):
            vec = vectors.get(token)
            if vec is None:
                continue
            score = association_score(vec, a_vecs, b_vecs)
            if not math.isfinite(score):
                continue
            female_scores.append(score)
            word_rows.append(
                {
                    "attribute_set": set_name,
                    "target_group": "female",
                    "token": token,
                    "score_s_w_A_B": score,
                }
            )

        male_mean = safe_mean(male_scores)
        female_mean = safe_mean(female_scores)
        diff = male_mean - female_mean if math.isfinite(male_mean) and math.isfinite(female_mean) else float("nan")
        pooled = male_scores + female_scores
        pooled_std = safe_std(pooled)
        effect_size = diff / pooled_std if math.isfinite(diff) and math.isfinite(pooled_std) and pooled_std != 0 else float("nan")

        summary_rows.append(
            {
                "attribute_set": set_name,
                "description": block["description"],
                "task_filter": args.task,
                "language_filter": args.language,
                "model_filter": args.model,
                "male_target_n": len(male_scores),
                "female_target_n": len(female_scores),
                "attr_A_used_n": len(a_words),
                "attr_B_used_n": len(b_words),
                "male_mean_s_w_A_B": male_mean,
                "female_mean_s_w_A_B": female_mean,
                "mean_diff_male_minus_female": diff,
                "effect_size_cohen_d": effect_size,
                "interpretation": "positive => male targets more associated with A than B (relative to female targets)"
                if math.isfinite(diff) and diff > 0
                else "negative => female targets more associated with A than B (relative to male targets)",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "weat_summary.csv"
    word_path = args.output_dir / "weat_word_scores.csv"
    oov_path = args.output_dir / "weat_oov.json"

    write_csv(
        summary_path,
        summary_rows,
        [
            "attribute_set",
            "description",
            "task_filter",
            "language_filter",
            "model_filter",
            "male_target_n",
            "female_target_n",
            "attr_A_used_n",
            "attr_B_used_n",
            "male_mean_s_w_A_B",
            "female_mean_s_w_A_B",
            "mean_diff_male_minus_female",
            "effect_size_cohen_d",
            "interpretation",
        ],
    )
    write_csv(word_path, word_rows, ["attribute_set", "target_group", "token", "score_s_w_A_B"])
    oov_path.write_text(
        json.dumps(
            {
                "missing_target_words": sorted(set(missing_targets)),
                "missing_count": len(set(missing_targets)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {summary_path.as_posix()} (rows={len(summary_rows)})")
    print(f"Wrote: {word_path.as_posix()} (rows={len(word_rows)})")
    print(f"Wrote: {oov_path.as_posix()}")
    print(f"[edit attributes] {args.attributes_file.as_posix()}")
    print(f"[embedding backend] {backend_name}: {embedding_file.as_posix()}")


if __name__ == "__main__":
    main()
