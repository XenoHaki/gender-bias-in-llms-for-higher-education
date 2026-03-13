from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


INPUT_ROOT = Path("data/results_resume/scoring")
OUTPUT_ROOT = Path("analysis/modeling/catboost_resume_by_model")
TRAIN_SCRIPT = Path("analysis/train_catboost_resume.py")
DISCOVER_BAR_COLOR = "cyan"
RUN_BAR_COLOR = "green"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one CatBoost resume model per LLM model."
    )
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--train-script", type=Path, default=TRAIN_SCRIPT)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--language", choices=["all", "zh", "en"], default="all")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Optional exact model names to run. Repeatable.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip model if output metrics.json already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--bagging-temperature", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--verbose-eval", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--shap-sample-size", type=int, default=50000)
    parser.add_argument("--save-shap-rows", action="store_true")
    return parser.parse_args()


def infer_model_name(rec: Dict, path: Path) -> str:
    model = str(rec.get("model") or "").strip()
    if model:
        return model
    key = str(rec.get("model_key") or "").strip()
    if key:
        return key
    parts = path.parts
    if "scoring" in parts:
        idx = parts.index("scoring")
        if idx + 2 < len(parts):
            return f"{parts[idx+1]}/{parts[idx+2]}"
    return "unknown"


def discover_models(root: Path, language: str) -> Counter:
    counts: Counter = Counter()
    files = sorted(root.rglob("standard.jsonl"))
    for path in tqdm(files, desc="discover models", unit="file", colour=DISCOVER_BAR_COLOR):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except json.JSONDecodeError:
                    continue
                lang = str(rec.get("language") or "").strip().lower()
                if language != "all" and lang != language:
                    continue
                model_name = infer_model_name(rec, path)
                counts[model_name] += 1
    return counts


def slugify_model_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")
    return slug or "unknown"


def build_train_command(
    args: argparse.Namespace,
    model_name: str,
    output_dir: Path,
) -> List[str]:
    cmd = [
        str(args.python_exe),
        str(args.train_script),
        "--input-root",
        str(args.input_root),
        "--output-dir",
        str(output_dir),
        "--language",
        args.language,
        "--model",
        model_name,
        "--exact-model-filter",
        "--test-size",
        str(args.test_size),
        "--seed",
        str(args.seed),
        "--max-rows",
        str(args.max_rows),
        "--iterations",
        str(args.iterations),
        "--learning-rate",
        str(args.learning_rate),
        "--depth",
        str(args.depth),
        "--l2-leaf-reg",
        str(args.l2_leaf_reg),
        "--random-strength",
        str(args.random_strength),
        "--bagging-temperature",
        str(args.bagging_temperature),
        "--early-stopping-rounds",
        str(args.early_stopping_rounds),
        "--verbose-eval",
        str(args.verbose_eval),
        "--cv-folds",
        str(args.cv_folds),
        "--shap-sample-size",
        str(args.shap_sample_size),
    ]
    if args.save_shap_rows:
        cmd.append("--save-shap-rows")
    return cmd


def select_models(counts: Counter, only_models: List[str] | None) -> List[Tuple[str, int]]:
    if not only_models:
        return sorted(counts.items(), key=lambda kv: kv[0].lower())
    only_set = set(only_models)
    out = [(m, counts[m]) for m in sorted(only_set, key=lambda x: x.lower()) if m in counts]
    missing = [m for m in only_models if m not in counts]
    if missing:
        print("[warn] models not found in data:")
        for m in missing:
            print(f"  - {m}")
    return out


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    counts = discover_models(args.input_root, args.language)
    models = select_models(counts, args.model)
    if not models:
        raise SystemExit("No models selected.")

    slug_seen: Dict[str, int] = {}
    plan_rows = []
    for model_name, row_count in models:
        slug = slugify_model_name(model_name)
        slug_seen[slug] = slug_seen.get(slug, 0) + 1
        if slug_seen[slug] > 1:
            slug = f"{slug}-{slug_seen[slug]}"
        out_dir = args.output_root / slug
        plan_rows.append((model_name, row_count, out_dir))

    print(f"[plan] selected models: {len(plan_rows)}")
    for model_name, row_count, out_dir in plan_rows:
        print(f"  - {model_name}: rows={row_count}, output={out_dir.as_posix()}")

    if args.dry_run:
        print("[dry-run] commands:")
        for model_name, _, out_dir in plan_rows:
            cmd = build_train_command(args, model_name, out_dir)
            print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
        return

    failed = []
    for model_name, row_count, out_dir in tqdm(
        plan_rows, desc="train per model", unit="model", colour=RUN_BAR_COLOR
    ):
        metrics_path = out_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_train_command(args, model_name, out_dir)
        print(f"\n[run] model={model_name} rows={row_count}")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            failed.append(model_name)
            print(f"[error] model failed: {model_name}")

    if failed:
        print("[done] failed models:")
        for model_name in failed:
            print(f"  - {model_name}")
        raise SystemExit(1)

    print("[done] all selected models finished")


if __name__ == "__main__":
    main()
