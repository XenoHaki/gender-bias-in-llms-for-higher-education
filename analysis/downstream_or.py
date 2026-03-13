from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable

import typer

app = typer.Typer(add_completion=False, help="Compute odds-ratio statistics for downstream adjectives.")


def iter_records(path: Path) -> Iterable[Dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@app.command()
def run(
    adjectives_path: Path = typer.Option(
        Path("analysis/downstream_adjectives.jsonl"), exists=True, help="Input adjective jsonl from extraction."
    ),
    output_path: Path = typer.Option(
        Path("analysis/downstream_or.csv"), help="CSV to store odds ratios per adjective/model."
    ),
    min_count: int = typer.Option(3, help="Minimum total count across persons to include a term."),
) -> None:
    by_model = defaultdict(lambda: {"xiaogang": Counter(), "xiaoting": Counter()})
    totals = defaultdict(lambda: {"xiaogang": 0, "xiaoting": 0})

    for record in iter_records(adjectives_path):
        model = record.get("model") or record.get("model_key")
        person = record.get("person")
        adjectives = record.get("adjectives") or []
        if model is None or person not in {"xiaogang", "xiaoting"}:
            continue
        by_model[model][person].update(adjectives)
        totals[model][person] += len(adjectives)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("model,adjective,count_xiaogang,count_xiaoting,odds_ratio\n")
        for model, counters in by_model.items():
            male_total = totals[model]["xiaogang"]
            female_total = totals[model]["xiaoting"]
            vocab = set(counters["xiaogang"]) | set(counters["xiaoting"])
            for word in sorted(vocab):
                male_count = counters["xiaogang"][word]
                female_count = counters["xiaoting"][word]
                if male_count + female_count < min_count:
                    continue
                odds_xg = (male_count + 0.5) / (male_total - male_count + 0.5)
                odds_xt = (female_count + 0.5) / (female_total - female_count + 0.5)
                odds_ratio = odds_xg / odds_xt if odds_xt else float("inf")
                fh.write(f"{model},{word},{male_count},{female_count},{odds_ratio:.6f}\n")
    typer.echo(f"Wrote odds-ratio table to {output_path}")


if __name__ == "__main__":
    app()
