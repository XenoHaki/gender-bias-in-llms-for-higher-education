from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

try:
    from catboost import CatBoostRegressor, Pool
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'catboost'. Install with `pip install catboost`.") from exc

from train_catboost_resume import (
    INPUT_ROOT,
    infer_model_name,
    infer_gender,
    load_dataset,
    split_indices,
)


DEFAULT_MODEL_DIR = Path("analysis/modeling/catboost_resume")
DEFAULT_OUTPUT_DIR = Path("analysis/modeling/catboost_resume_interaction")
WRITE_BAR_COLOR = "green"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute interaction SHAP for resume-scoring CatBoost model.")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--language", choices=["all", "zh", "en"], default="all")
    parser.add_argument("--model", action="append", default=None, help="Model substring filter, repeatable.")
    parser.add_argument("--exact-model-filter", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of test rows for interaction SHAP. 0=all.")
    parser.add_argument("--batch-size", type=int, default=200, help="Rows per interaction SHAP batch.")
    return parser.parse_args()


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row", colour=WRITE_BAR_COLOR):
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_dir / "catboost_resume_model.cbm"
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    print("[1/6] loading dataset...", flush=True)
    x, y, encoder_dump, feature_names, gender_codes = load_dataset(
        root=args.input_root,
        language=args.language,
        model_filters=args.model,
        exact_model_filter=args.exact_model_filter,
        max_rows=0,
    )
    print(f"[data] rows={len(y)}, features={len(feature_names)}", flush=True)

    print("[2/6] rebuilding train/test split...", flush=True)
    train_idx, test_idx = split_indices(len(y), args.test_size, args.seed)
    x_test = x[test_idx]
    y_test = y[test_idx]
    gender_test = gender_codes[test_idx]
    print(f"[split] train={len(train_idx)}, test={len(test_idx)}", flush=True)

    if args.sample_size and len(x_test) > args.sample_size:
        rng = np.random.default_rng(args.seed)
        sample_idx = np.sort(rng.choice(len(x_test), size=args.sample_size, replace=False))
        x_used = x_test[sample_idx]
        y_used = y_test[sample_idx]
        gender_used = gender_test[sample_idx]
    else:
        x_used = x_test
        y_used = y_test
        gender_used = gender_test
    print(f"[sample] rows={len(x_used)}, batch_size={args.batch_size}", flush=True)

    print("[3/6] loading CatBoost model...", flush=True)
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    cat_features = list(range(8))

    print("[4/6] computing interaction SHAP...", flush=True)
    batches: List[np.ndarray] = []
    for start in tqdm(range(0, len(x_used), args.batch_size), desc="interaction shap", unit="batch", colour="cyan"):
        batch = x_used[start : start + args.batch_size]
        pool = Pool(batch, cat_features=cat_features)
        values = model.get_feature_importance(type="ShapInteractionValues", data=pool)
        batches.append(np.asarray(values, dtype=np.float64))
        done = min(start + args.batch_size, len(x_used))
        print(f"[progress] interaction rows processed: {done}/{len(x_used)}", flush=True)

    interaction = np.concatenate(batches, axis=0)
    # CatBoost returns (n_rows, n_features + 1, n_features + 1); last index is expected value.
    interaction = interaction[:, :-1, :-1]

    print("[5/6] aggregating summaries...", flush=True)
    abs_mean = np.mean(np.abs(interaction), axis=0)
    signed_mean = np.mean(interaction, axis=0)
    rows: List[Dict] = []
    for i, f1 in enumerate(feature_names):
        for j, f2 in enumerate(feature_names):
            rows.append(
                {
                    "feature_1": f1,
                    "feature_2": f2,
                    "mean_abs_interaction": float(abs_mean[i, j]),
                    "mean_interaction": float(signed_mean[i, j]),
                }
            )

    gender_idx = feature_names.index("gender")
    gender_rows: List[Dict] = []
    for j, feature in enumerate(feature_names):
        gender_rows.append(
            {
                "feature": feature,
                "mean_abs_interaction_with_gender": float(abs_mean[gender_idx, j]),
                "mean_interaction_with_gender": float(signed_mean[gender_idx, j]),
            }
        )

    inv_gender = {v: k for k, v in encoder_dump["gender"].items()}
    by_gender_rows: List[Dict] = []
    for code in sorted(np.unique(gender_used)):
        mask = gender_used == code
        if not np.any(mask):
            continue
        block_abs = np.mean(np.abs(interaction[mask]), axis=0)
        block_signed = np.mean(interaction[mask], axis=0)
        label = inv_gender.get(int(code), str(code))
        for j, feature in enumerate(feature_names):
            by_gender_rows.append(
                {
                    "gender": label,
                    "feature": feature,
                    "mean_abs_interaction_with_gender": float(block_abs[gender_idx, j]),
                    "mean_interaction_with_gender": float(block_signed[gender_idx, j]),
                    "n_rows": int(mask.sum()),
                }
            )

    top_pairs = sorted(
        (
            row
            for row in rows
            if row["feature_1"] <= row["feature_2"]
        ),
        key=lambda x: x["mean_abs_interaction"],
        reverse=True,
    )[:20]

    report = {
        "model_path": model_path.as_posix(),
        "input_root": args.input_root.as_posix(),
        "language": args.language,
        "model_filter": args.model or [],
        "exact_model_filter": bool(args.exact_model_filter),
        "test_size": args.test_size,
        "seed": args.seed,
        "sample_size_requested": args.sample_size,
        "sample_size_used": int(len(x_used)),
        "feature_names": feature_names,
        "top_interaction_pairs": top_pairs,
    }

    print("[6/6] writing outputs...", flush=True)
    write_csv(
        args.output_dir / "interaction_shap_summary.csv",
        rows,
        ["feature_1", "feature_2", "mean_abs_interaction", "mean_interaction"],
    )
    write_csv(
        args.output_dir / "gender_interaction_shap_summary.csv",
        gender_rows,
        ["feature", "mean_abs_interaction_with_gender", "mean_interaction_with_gender"],
    )
    write_csv(
        args.output_dir / "gender_interaction_shap_by_gender.csv",
        by_gender_rows,
        ["gender", "feature", "mean_abs_interaction_with_gender", "mean_interaction_with_gender", "n_rows"],
    )
    (args.output_dir / "interaction_shap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] output_dir={args.output_dir.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
