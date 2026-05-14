from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


INPUT_ROOT = Path("data/results_resume/scoring")
OUTPUT_ROOT = Path("analysis/modeling/catboost_resume_by_major")
TRAIN_SCRIPT = Path("analysis/train_catboost_resume.py")
DISCOVER_BAR_COLOR = "cyan"
RUN_BAR_COLOR = "green"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one CatBoost resume model per major.")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--train-script", type=Path, default=TRAIN_SCRIPT)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--language", choices=["all", "zh", "en"], default="all")
    parser.add_argument("--major", action="append", default=None, help="Optional exact major names to run.")
    parser.add_argument("--skip-existing", action="store_true")
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


def discover_majors(root: Path, language: str) -> Counter:
    counts: Counter = Counter()
    files = sorted(root.rglob("standard.jsonl"))
    for path in tqdm(files, desc="discover majors", unit="file", colour=DISCOVER_BAR_COLOR):
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
                major = str(rec.get("major") or "").strip()
                if major:
                    counts[major] += 1
    return counts


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")
    if slug:
        return slug
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:10]
    return f"major-{digest}"


def build_train_command(args: argparse.Namespace, major: str, output_dir: Path) -> List[str]:
    cmd = [
        str(args.python_exe),
        str(args.train_script),
        "--input-root",
        str(args.input_root),
        "--output-dir",
        str(output_dir),
        "--language",
        args.language,
        "--major",
        major,
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


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    counts = discover_majors(args.input_root, args.language)
    majors = sorted(counts.items(), key=lambda kv: kv[0])
    if args.major:
        wanted = set(args.major)
        majors = [(major, n) for major, n in majors if major in wanted]
    if not majors:
        raise SystemExit("No majors selected.")

    print(f"[plan] selected majors: {len(majors)}")
    for major, rows in majors:
        print(f"  - {major}: rows={rows}, output={(args.output_root / slugify(major)).as_posix()}")

    if args.dry_run:
        for major, _ in majors:
            out_dir = args.output_root / slugify(major)
            print(" ".join(build_train_command(args, major, out_dir)))
        return

    failed: List[str] = []
    for major, rows in tqdm(majors, desc="train per major", unit="major", colour=RUN_BAR_COLOR):
        out_dir = args.output_root / slugify(major)
        metrics_path = out_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_train_command(args, major, out_dir)
        print(f"\n[run] major={major} rows={rows}", flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            failed.append(major)
            print(f"[error] major failed: {major}", flush=True)

    if failed:
        print("[done] failed majors:")
        for major in failed:
            print(f"  - {major}")
        raise SystemExit(1)
    print("[done] all selected majors finished")


if __name__ == "__main__":
    main()
