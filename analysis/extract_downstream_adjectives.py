from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import typer

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

app = typer.Typer(add_completion=False, help="Extract adjectives from downstream recommendation letters.")


class AdjectiveExtractor:
    def __init__(self, english_model: str = "en_core_web_sm", chinese_model: str = "zh_core_web_sm") -> None:
        self.english_model_name = english_model
        self.chinese_model_name = chinese_model
        self._spacy_en = None
        self._spacy_zh = None

    def _load_model(self, model_name: str):
        if spacy is None:
            raise RuntimeError("spaCy is not installed. Run `pip install spacy` and download required models.")
        try:
            return spacy.load(model_name)
        except OSError as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load spaCy model '{model_name}'. Install via `python -m spacy download {model_name}`."
            ) from exc

    @property
    def nlp_en(self):
        if self._spacy_en is None:
            self._spacy_en = self._load_model(self.english_model_name)
        return self._spacy_en

    @property
    def nlp_zh(self):
        if self._spacy_zh is None:
            self._spacy_zh = self._load_model(self.chinese_model_name)
        return self._spacy_zh

    def extract(self, text: str, language: str) -> List[str]:
        if not text:
            return []
        lang = (language or "zh").lower()
        if lang.startswith("en"):
            doc = self.nlp_en(text)
        else:
            doc = self.nlp_zh(text)
        return [token.text for token in doc if token.pos_ == "ADJ"]


def iter_records(base_dir: Path) -> Iterable[Dict]:
    for path in base_dir.rglob("standard.jsonl"):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


@app.command()
def run(
    base_dir: Path = typer.Option(Path("data/results_downstream"), exists=True, help="Input directory with jsonl files."),
    output_path: Path = typer.Option(Path("analysis/downstream_adjectives.jsonl"), help="Where to write adjective records."),
    english_model: str = typer.Option("en_core_web_sm", help="spaCy English model name."),
    chinese_model: str = typer.Option("zh_core_web_sm", help="spaCy Chinese model name."),
    default_language: str = typer.Option("zh", help="Fallback language tag when missing in records."),
) -> None:
    extractor = AdjectiveExtractor(english_model=english_model, chinese_model=chinese_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for record in iter_records(base_dir):
            response = record.get("response", "")
            language = record.get("language") or default_language
            adjectives = extractor.extract(response, language)
            payload = {
                "model": record.get("model"),
                "model_key": record.get("model_key"),
                "major": record.get("major"),
                "person": record.get("person"),
                "language": language,
                "run_index": record.get("run_index"),
                "adjectives": adjectives,
                "text_length": len(response),
            }
            out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            processed += 1
    typer.echo(f"Extracted adjectives for {processed} recommendation letters -> {output_path}")


if __name__ == "__main__":
    app()
