from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import typer
from sentence_transformers import SentenceTransformer

app = typer.Typer(add_completion=False, help="Compute WEAT-style metrics for downstream adjectives.")

TARGET_WORDS = {"xiaogang": "小刚", "xiaoting": "小婷"}


def iter_records(path: Path) -> Iterable[Dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def embed_terms(model: SentenceTransformer, terms: List[str]) -> Dict[str, np.ndarray]:
    vectors = model.encode(terms, normalize_embeddings=True)
    return {term: vec for term, vec in zip(terms, vectors)}


def compute_effect(
    embeddings: Dict[str, np.ndarray],
    male_vec: np.ndarray,
    female_vec: np.ndarray,
    male_counts: Counter,
    female_counts: Counter,
) -> Dict:
    male_scores: List[float] = []
    female_scores: List[float] = []
    for word, count in male_counts.items():
        vec = embeddings.get(word)
        if vec is None:
            continue
        score = float(np.dot(vec, male_vec) - np.dot(vec, female_vec))
        male_scores.extend([score] * count)
    for word, count in female_counts.items():
        vec = embeddings.get(word)
        if vec is None:
            continue
        score = float(np.dot(vec, male_vec) - np.dot(vec, female_vec))
        female_scores.extend([score] * count)

    if not male_scores or not female_scores:
        return {"effect_size": float("nan"), "mean_diff": float("nan")}

    male_mean = float(np.mean(male_scores))
    female_mean = float(np.mean(female_scores))
    combined = np.array(male_scores + female_scores)
    std_dev = float(np.std(combined, ddof=1)) or math.inf
    effect_size = (male_mean - female_mean) / std_dev
    return {"effect_size": effect_size, "mean_diff": male_mean - female_mean}


@app.command()
def run(
    adjectives_path: Path = typer.Option(
        Path("analysis/downstream_adjectives.jsonl"), exists=True, help="Input adjective jsonl from extraction."
    ),
    output_path: Path = typer.Option(Path("analysis/downstream_weat.csv"), help="CSV summary path."),
    embedding_model: str = typer.Option(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformers model name."
    ),
) -> None:
    by_model = defaultdict(lambda: {"xiaogang": Counter(), "xiaoting": Counter()})
    for record in iter_records(adjectives_path):
        model_name = record.get("model") or record.get("model_key")
        person = record.get("person")
        adjectives = record.get("adjectives") or []
        if model_name is None or person not in {"xiaogang", "xiaoting"}:
            continue
        by_model[model_name][person].update(adjectives)

    embedder = SentenceTransformer(embedding_model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("model,effect_size,mean_diff\n")
        for model_name, counters in by_model.items():
            vocab = list(set(counters["xiaogang"]) | set(counters["xiaoting"]))
            terms = vocab + list(TARGET_WORDS.values())
            embeddings = embed_terms(embedder, terms)
            male_vec = embeddings.get(TARGET_WORDS["xiaogang"])
            female_vec = embeddings.get(TARGET_WORDS["xiaoting"])
            if male_vec is None or female_vec is None:
                typer.echo(
                    f"Skipping {model_name} due to missing embeddings for target words. Check model vocabulary.",
                    err=True,
                )
                continue
            stats = compute_effect(embeddings, male_vec, female_vec, counters["xiaogang"], counters["xiaoting"])
            fh.write(f"{model_name},{stats['effect_size']},{stats['mean_diff']}\n")

    typer.echo(f"Wrote WEAT summary to {output_path}")


if __name__ == "__main__":
    app()
