from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from catboost import CatBoostRegressor, Pool
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'catboost'. Install with `pip install catboost`.") from exc


INPUT_ROOT = Path("data/results_resume/scoring")
OUTPUT_DIR = Path("analysis/modeling/catboost_resume")
LOAD_BAR_COLOR = "cyan"
WRITE_BAR_COLOR = "green"
CV_BAR_COLOR = "magenta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost on downstream resume scoring data.")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT, help="Root of scoring JSONL files.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for model/artifacts.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio in (0,1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for quick experiments; 0=all.")
    parser.add_argument("--language", choices=["all", "zh", "en"], default="all", help="Language filter.")
    parser.add_argument("--model", action="append", default=None, help="Model substring filter, repeatable.")
    parser.add_argument(
        "--major",
        action="append",
        default=None,
        help="Exact major filter, repeatable. Use language-consistent major strings.",
    )
    parser.add_argument(
        "--exact-model-filter",
        action="store_true",
        help="Match --model values exactly (case-insensitive) instead of substring contains.",
    )
    parser.add_argument("--iterations", type=int, default=1200, help="Max boosting rounds.")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--bagging-temperature", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--verbose-eval", type=int, default=100)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="K-fold CV folds. 0 or 1 disables CV; e.g., 5 for 5-fold.",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=50000,
        help="Number of test rows used for SHAP contribution summary; 0=all test rows.",
    )
    parser.add_argument(
        "--save-shap-rows",
        action="store_true",
        help="Save sampled per-row SHAP contributions (wide CSV, large file).",
    )
    return parser.parse_args()


def parse_score(rec: Dict) -> Optional[float]:
    score = rec.get("score")
    if isinstance(score, (int, float)):
        value = float(score)
        if 0 <= value <= 100:
            return value

    response = str(rec.get("score_response") or "").strip()
    if not response:
        return None
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", response)
    for token in reversed(nums):
        try:
            value = float(token)
        except ValueError:
            continue
        if 0 <= value <= 100:
            return value
    return None


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


def infer_gender(rec: Dict) -> str:
    gid = str(rec.get("gender_id") or "").strip().lower()
    if gid in {"male", "female"}:
        return gid
    person = str(rec.get("person_id") or "").strip().lower()
    if person == "xiaogang":
        return "male"
    if person == "xiaoting":
        return "female"
    g = str(rec.get("gender") or "").strip().lower()
    if g in {"male", "\u7537"}:
        return "male"
    if g in {"female", "\u5973"}:
        return "female"
    return "unknown"


class LabelEncoder:
    def __init__(self) -> None:
        self.mapping: Dict[str, int] = {}

    def encode(self, value: str) -> int:
        key = (value or "").strip()
        if key not in self.mapping:
            self.mapping[key] = len(self.mapping)
        return self.mapping[key]

    def dump(self) -> Dict[str, int]:
        return dict(self.mapping)


def load_dataset(
    root: Path,
    language: str,
    model_filters: Optional[List[str]],
    exact_model_filter: bool,
    major_filters: Optional[List[str]],
    max_rows: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, int]], List[str], np.ndarray]:
    feature_names = [
        "model",
        "language",
        "gender",
        "major",
        "gpa",
        "competition",
        "internship",
        "english",
        "measurement",
    ]

    encoders = {
        "model": LabelEncoder(),
        "language": LabelEncoder(),
        "gender": LabelEncoder(),
        "major": LabelEncoder(),
        "gpa": LabelEncoder(),
        "competition": LabelEncoder(),
        "internship": LabelEncoder(),
        "english": LabelEncoder(),
    }

    x_rows: List[List[int]] = []
    y_vals: List[float] = []
    gender_codes: List[int] = []

    model_filters_lower = [x.lower() for x in (model_filters or [])]
    major_filter_set = {x.strip() for x in (major_filters or []) if x and x.strip()}
    files = sorted(root.rglob("standard.jsonl"))
    for path in tqdm(files, desc="load scoring files", unit="file", colour=LOAD_BAR_COLOR):
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
                if model_filters_lower:
                    model_name_l = model_name.lower()
                    if exact_model_filter:
                        if model_name_l not in model_filters_lower:
                            continue
                    elif not any(x in model_name_l for x in model_filters_lower):
                        continue

                score = parse_score(rec)
                if score is None:
                    continue

                gender = infer_gender(rec)
                major = str(rec.get("major") or "").strip()
                if major_filter_set and major not in major_filter_set:
                    continue
                gpa = str(rec.get("gpa") or "").strip()
                competition = str(rec.get("competition") or "").strip()
                internship = str(rec.get("internship") or "").strip()
                english_score = str(rec.get("english") or "").strip()

                measurement_raw = rec.get("measurement")
                if isinstance(measurement_raw, (int, float)):
                    measurement = int(measurement_raw)
                else:
                    measurement = 0

                row = [
                    encoders["model"].encode(model_name),
                    encoders["language"].encode(lang),
                    encoders["gender"].encode(gender),
                    encoders["major"].encode(major),
                    encoders["gpa"].encode(gpa),
                    encoders["competition"].encode(competition),
                    encoders["internship"].encode(internship),
                    encoders["english"].encode(english_score),
                    measurement,
                ]
                x_rows.append(row)
                y_vals.append(score)
                gender_codes.append(row[2])

                if max_rows and len(y_vals) >= max_rows:
                    break
        if max_rows and len(y_vals) >= max_rows:
            break

    if not y_vals:
        raise SystemExit("No usable rows after filtering.")

    x = np.asarray(x_rows, dtype=np.int32)
    y = np.asarray(y_vals, dtype=np.float32)
    gender_arr = np.asarray(gender_codes, dtype=np.int32)
    encoder_dump = {k: enc.dump() for k, enc in encoders.items()}
    return x, y, encoder_dump, feature_names, gender_arr


def split_indices(n: int, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise SystemExit("Need at least 2 rows to split train/test.")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    test_n = max(1, int(n * test_size))
    if test_n >= n:
        test_n = n - 1
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return train_idx, test_idx


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    mean_true = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def subset_metrics_by_gender(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gender_codes: np.ndarray,
    gender_mapping: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, code in gender_mapping.items():
        mask = gender_codes == code
        if not np.any(mask):
            continue
        out[name] = regression_metrics(y_true[mask], y_pred[mask])
        out[name]["n"] = int(np.sum(mask))
    return out


def write_feature_importance(path: Path, feature_names: List[str], importance: np.ndarray) -> None:
    rows = []
    for i, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "importance": float(importance[i]),
            }
        )
    rows.sort(key=lambda x: x["importance"], reverse=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["feature", "importance"])
        writer.writeheader()
        writer.writerows(rows)


def invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, code in mapping.items():
        out[int(code)] = name
    return out


def write_shap_summary(path: Path, feature_names: List[str], contrib: np.ndarray) -> None:
    mean_abs = np.mean(np.abs(contrib), axis=0)
    mean_signed = np.mean(contrib, axis=0)
    rows = []
    for i, feature in enumerate(feature_names):
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(mean_abs[i]),
                "mean_shap": float(mean_signed[i]),
            }
        )
    rows.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["feature", "mean_abs_shap", "mean_shap"])
        writer.writeheader()
        writer.writerows(rows)


def write_shap_by_gender(
    path: Path,
    feature_names: List[str],
    contrib: np.ndarray,
    gender_codes: np.ndarray,
    gender_label_by_code: Dict[int, str],
) -> None:
    rows = []
    for code in sorted(set(int(x) for x in gender_codes.tolist())):
        mask = gender_codes == code
        if not np.any(mask):
            continue
        label = gender_label_by_code.get(code, f"code_{code}")
        part = contrib[mask]
        mean_abs = np.mean(np.abs(part), axis=0)
        mean_signed = np.mean(part, axis=0)
        for i, feature in enumerate(feature_names):
            rows.append(
                {
                    "gender": label,
                    "n": int(np.sum(mask)),
                    "feature": feature,
                    "mean_abs_shap": float(mean_abs[i]),
                    "mean_shap": float(mean_signed[i]),
                }
            )
    rows.sort(key=lambda x: (x["gender"], -x["mean_abs_shap"]))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["gender", "n", "feature", "mean_abs_shap", "mean_shap"])
        writer.writeheader()
        writer.writerows(rows)


def write_shap_rows(
    path: Path,
    feature_names: List[str],
    contrib: np.ndarray,
    expected_value: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gender_codes: np.ndarray,
    gender_label_by_code: Dict[int, str],
) -> None:
    fields = ["row_index", "y_true", "y_pred", "expected_value", "gender"]
    fields.extend(feature_names)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in tqdm(range(contrib.shape[0]), desc="write shap rows", unit="row", colour=WRITE_BAR_COLOR):
            row = {
                "row_index": i,
                "y_true": float(y_true[i]),
                "y_pred": float(y_pred[i]),
                "expected_value": float(expected_value[i]),
                "gender": gender_label_by_code.get(int(gender_codes[i]), f"code_{int(gender_codes[i])}"),
            }
            for j, feature in enumerate(feature_names):
                row[feature] = float(contrib[i, j])
            writer.writerow(row)


def build_kfold_indices(n: int, folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        return []
    if folds > n:
        folds = n
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = np.array_split(idx, folds)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(folds):
        valid_idx = split[i]
        train_parts = [split[j] for j in range(folds) if j != i]
        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
        out.append((train_idx, valid_idx))
    return out


def build_pool(x: np.ndarray, y: np.ndarray, cat_features: List[int]) -> Pool:
    return Pool(data=x, label=y, cat_features=cat_features)


def build_model(args: argparse.Namespace) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_strength=args.random_strength,
        bagging_temperature=args.bagging_temperature,
        random_seed=args.seed,
        verbose=args.verbose_eval,
    )


def run_kfold_cv(
    x: np.ndarray,
    y: np.ndarray,
    gender_codes: np.ndarray,
    gender_mapping: Dict[str, int],
    cat_features: List[int],
    args: argparse.Namespace,
) -> Dict[str, object]:
    split = build_kfold_indices(len(y), folds=args.cv_folds, seed=args.seed)
    if not split:
        return {}

    fold_rows: List[Dict[str, float]] = []
    for fold_id, (train_idx, valid_idx) in enumerate(
        tqdm(split, desc=f"{args.cv_folds}-fold CV", unit="fold", colour=CV_BAR_COLOR), start=1
    ):
        x_tr, y_tr = x[train_idx], y[train_idx]
        x_va, y_va = x[valid_idx], y[valid_idx]
        _ = gender_codes, gender_mapping

        model = build_model(args)
        model.fit(
            build_pool(x_tr, y_tr, cat_features),
            eval_set=build_pool(x_va, y_va, cat_features),
            use_best_model=True,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        pred = model.predict(x_va)
        m = regression_metrics(y_va, pred)
        fold_rows.append(
            {
                "fold": float(fold_id),
                "train_n": float(len(train_idx)),
                "valid_n": float(len(valid_idx)),
                "best_iteration": float(model.get_best_iteration()),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
            }
        )

    mae_vals = [r["mae"] for r in fold_rows]
    rmse_vals = [r["rmse"] for r in fold_rows]
    r2_vals = [r["r2"] for r in fold_rows]
    best_iters = [r["best_iteration"] for r in fold_rows]
    summary = {
        "folds": args.cv_folds,
        "mae_mean": float(np.mean(mae_vals)),
        "mae_std": float(np.std(mae_vals, ddof=1)) if len(mae_vals) > 1 else 0.0,
        "rmse_mean": float(np.mean(rmse_vals)),
        "rmse_std": float(np.std(rmse_vals, ddof=1)) if len(rmse_vals) > 1 else 0.0,
        "r2_mean": float(np.mean(r2_vals)),
        "r2_std": float(np.std(r2_vals, ddof=1)) if len(r2_vals) > 1 else 0.0,
        "best_iteration_mean": float(np.mean(best_iters)),
        "best_iteration_std": float(np.std(best_iters, ddof=1)) if len(best_iters) > 1 else 0.0,
    }
    return {"summary": summary, "fold_metrics": fold_rows}


def write_cv_metrics(path: Path, fold_rows: List[Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["fold", "train_n", "valid_n", "best_iteration", "mae", "rmse", "r2"],
        )
        writer.writeheader()
        writer.writerows(fold_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, y, encoder_dump, feature_names, gender_codes = load_dataset(
        root=args.input_root,
        language=args.language,
        model_filters=args.model,
        exact_model_filter=args.exact_model_filter,
        major_filters=args.major,
        max_rows=args.max_rows,
    )
    print(f"[data] rows={len(y)}, features={x.shape[1]}")

    train_idx, test_idx = split_indices(len(y), test_size=args.test_size, seed=args.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    test_gender = gender_codes[test_idx]
    print(f"[split] train={len(train_idx)}, test={len(test_idx)}")

    cat_features = list(range(8))
    train_pool = build_pool(x_train, y_train, cat_features)
    test_pool = build_pool(x_test, y_test, cat_features)

    cv_result: Dict[str, object] = {}
    cv_path = args.output_dir / "cv_fold_metrics.csv"
    if args.cv_folds and args.cv_folds > 1:
        print(f"[cv] start {args.cv_folds}-fold cross validation")
        cv_result = run_kfold_cv(
            x=x,
            y=y,
            gender_codes=gender_codes,
            gender_mapping=encoder_dump["gender"],
            cat_features=cat_features,
            args=args,
        )
        fold_rows = cv_result.get("fold_metrics", [])
        if isinstance(fold_rows, list) and fold_rows:
            write_cv_metrics(cv_path, fold_rows)  # type: ignore[arg-type]
            print(f"[cv] fold metrics: {cv_path.as_posix()}")

    print("[train] start CatBoost")
    model = build_model(args)
    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    y_pred = np.asarray(model.predict(x_test), dtype=np.float32)
    overall = regression_metrics(y_test, y_pred)
    by_gender = subset_metrics_by_gender(
        y_true=y_test,
        y_pred=y_pred,
        gender_codes=test_gender,
        gender_mapping=encoder_dump["gender"],
    )

    model_path = args.output_dir / "catboost_resume_model.cbm"
    metrics_path = args.output_dir / "metrics.json"
    mapping_path = args.output_dir / "category_mappings.json"
    fi_path = args.output_dir / "feature_importance.csv"
    shap_summary_path = args.output_dir / "shap_summary.csv"
    shap_gender_path = args.output_dir / "shap_summary_by_gender.csv"
    shap_rows_path = args.output_dir / "shap_sample_rows.csv"

    model.save_model(str(model_path))
    importance = np.asarray(
        model.get_feature_importance(train_pool, type="PredictionValuesChange"),
        dtype=np.float64,
    )
    write_feature_importance(fi_path, feature_names, importance)

    if args.shap_sample_size and args.shap_sample_size > 0 and args.shap_sample_size < len(x_test):
        rng = np.random.default_rng(args.seed)
        shap_idx = rng.choice(len(x_test), size=args.shap_sample_size, replace=False)
        x_shap = x_test[shap_idx]
        y_shap_true = y_test[shap_idx]
        y_shap_pred = y_pred[shap_idx]
        gender_shap = test_gender[shap_idx]
    else:
        x_shap = x_test
        y_shap_true = y_test
        y_shap_pred = y_pred
        gender_shap = test_gender

    print(f"[shap] rows={len(x_shap)}")
    shap_values_all = np.asarray(
        model.get_feature_importance(build_pool(x_shap, y_shap_true, cat_features), type="ShapValues"),
        dtype=np.float64,
    )
    feature_contrib = shap_values_all[:, :-1]
    expected_value = shap_values_all[:, -1]

    gender_label_by_code = invert_mapping(encoder_dump["gender"])
    write_shap_summary(shap_summary_path, feature_names, feature_contrib)
    write_shap_by_gender(
        shap_gender_path,
        feature_names,
        feature_contrib,
        gender_shap,
        gender_label_by_code,
    )
    if args.save_shap_rows:
        write_shap_rows(
            shap_rows_path,
            feature_names,
            feature_contrib,
            expected_value,
            y_shap_true,
            y_shap_pred,
            gender_shap,
            gender_label_by_code,
        )

    report = {
        "rows": int(len(y)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "best_iteration": int(model.get_best_iteration()),
        "params": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "l2_leaf_reg": args.l2_leaf_reg,
            "random_strength": args.random_strength,
            "bagging_temperature": args.bagging_temperature,
            "seed": args.seed,
        },
        "metrics_overall": overall,
        "metrics_by_gender": by_gender,
        "metrics_cv": cv_result.get("summary", {}),
        "feature_names": feature_names,
        "shap_rows_used": int(len(x_shap)),
        "shap_summary_path": shap_summary_path.as_posix(),
        "shap_summary_by_gender_path": shap_gender_path.as_posix(),
        "shap_rows_path": shap_rows_path.as_posix() if args.save_shap_rows else None,
        "cv_fold_metrics_path": cv_path.as_posix() if cv_result else None,
    }
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping_path.write_text(json.dumps(encoder_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] model: {model_path.as_posix()}")
    print(f"[done] metrics: {metrics_path.as_posix()}")
    print(f"[done] mappings: {mapping_path.as_posix()}")
    print(f"[done] feature importance: {fi_path.as_posix()}")
    print(f"[done] shap summary: {shap_summary_path.as_posix()}")
    print(f"[done] shap by gender: {shap_gender_path.as_posix()}")
    if args.save_shap_rows:
        print(f"[done] shap rows: {shap_rows_path.as_posix()}")
    print(
        f"[metrics] MAE={overall['mae']:.4f}, RMSE={overall['rmse']:.4f}, R2={overall['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
