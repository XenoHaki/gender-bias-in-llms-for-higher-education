from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import jieba
import jieba.posseg as pseg
from tqdm import tqdm

try:
    import spacy
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'spacy'. Install it first, e.g. `pip install spacy`."
    ) from exc


MIDSTREAM_ROOT = Path("data/results_midstream")
DOWNSTREAM_ROOT = Path("data/results_downstream")
OUTPUT_DIR = Path("analysis/text_bias")
OVERALL_MODEL_LABEL = "__ALL_MODELS__"

EN_KEEP_POS = {"ADJ"}
ZH_KEEP_POS_HEAD = {"a"}
DEFAULT_BLACKLIST = {"小刚", "小婷", "bob", "mary", "xiaogang", "xiaoting"}


@dataclass(frozen=True)
class TermPattern:
    pieces: Tuple[str, ...]
    output: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Log-OR token bias for midstream/downstream text with Laplace smoothing."
    )
    parser.add_argument("--midstream-root", type=Path, default=MIDSTREAM_ROOT)
    parser.add_argument("--downstream-root", type=Path, default=DOWNSTREAM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha.")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum total count (M+F) per token.")
    parser.add_argument("--neutral-threshold", type=float, default=0.1, help="abs(Log-OR) below this is neutral.")
    parser.add_argument("--top-k", type=int, default=100, help="Top K male/female-biased tokens per group.")
    parser.add_argument(
        "--task",
        choices=["midstream", "downstream", "both"],
        default="both",
        help="Task scope to analyze.",
    )
    parser.add_argument(
        "--language",
        choices=["zh", "en", "all"],
        default="all",
        help="Language filter.",
    )
    parser.add_argument("--model", action="append", default=None, help="Model name filter (substring, repeatable).")
    parser.add_argument("--zh-terms", type=Path, default=None, help="Optional zh term lexicon (one per line).")
    parser.add_argument("--en-terms", type=Path, default=None, help="Optional en term lexicon (one per line).")
    parser.add_argument(
        "--blacklist",
        type=Path,
        default=None,
        help="Optional blacklist lexicon (one term per line).",
    )
    return parser.parse_args()


def _load_terms(path: Optional[Path], language: str) -> List[str]:
    if path is None or not path.exists():
        return []
    items: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line.lower() if language == "en" else line)
    return items


def _load_blacklist(path: Optional[Path]) -> set[str]:
    words = set(DEFAULT_BLACKLIST)
    if path is None or not path.exists():
        return words
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line)
    return words


def _tokenize_term(term: str, language: str) -> Tuple[str, ...]:
    if language == "en":
        parts = [x for x in re.split(r"\s+", term.lower()) if x]
        return tuple(parts)
    parts = [x.strip() for x in jieba.lcut(term) if x.strip()]
    return tuple(parts)


def _build_term_index(terms: Iterable[str], language: str) -> Dict[str, List[TermPattern]]:
    index: Dict[str, List[TermPattern]] = defaultdict(list)
    for term in terms:
        pieces = _tokenize_term(term, language)
        if not pieces:
            continue
        output = term.lower() if language == "en" else term
        index[pieces[0]].append(TermPattern(pieces=pieces, output=output))
    for key in index:
        index[key].sort(key=lambda x: len(x.pieces), reverse=True)
    return index


def _looks_valid_zh(word: str) -> bool:
    if not word:
        return False
    if re.fullmatch(r"[\W_]+", word, flags=re.UNICODE):
        return False
    return True


def _looks_valid_en(word: str) -> bool:
    if not word:
        return False
    if not re.search(r"[a-zA-Z]", word):
        return False
    return True


def _init_spacy_en():
    print("[init] Initializing spaCy English pipeline...", flush=True)
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
    except Exception:
        print("[init] spaCy model en_core_web_sm not found locally, downloading...", flush=True)
        from spacy.cli import download

        download("en_core_web_sm")
        print("[init] Download complete. Building pipeline...", flush=True)
        return spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])


class TokenExtractor:
    def __init__(self, zh_terms: List[str], en_terms: List[str], blacklist: set[str]):
        self.en_nlp = None
        self.zh_term_index = _build_term_index(zh_terms, "zh")
        self.en_term_index = _build_term_index(en_terms, "en")
        self.zh_term_set = {t for t in zh_terms}
        self.en_term_set = {t.lower() for t in en_terms}
        self.blacklist = {w.strip() for w in blacklist if w.strip()}
        self.blacklist_lower = {w.lower() for w in self.blacklist}

    def _ensure_en_nlp(self):
        if self.en_nlp is None:
            self.en_nlp = _init_spacy_en()
        return self.en_nlp

    def _is_blacklisted(self, token: str) -> bool:
        t = token.strip()
        if not t:
            return True
        return t in self.blacklist or t.lower() in self.blacklist_lower

    @staticmethod
    def _match_longest(
        tokens_norm: List[str], term_index: Dict[str, List[TermPattern]], start: int
    ) -> Optional[Tuple[int, str]]:
        first = tokens_norm[start]
        candidates = term_index.get(first, [])
        if not candidates:
            return None
        for cand in candidates:
            end = start + len(cand.pieces)
            if end > len(tokens_norm):
                continue
            if tuple(tokens_norm[start:end]) == cand.pieces:
                return len(cand.pieces), cand.output
        return None

    def extract(self, text: str, language: str) -> List[str]:
        if not text.strip():
            return []
        if language == "zh":
            return self._extract_zh(text)
        if language == "en":
            return self._extract_en(text)
        return []

    def _extract_zh(self, text: str) -> List[str]:
        base: List[Tuple[str, str, str]] = []
        for item in pseg.cut(text):
            word = (item.word or "").strip()
            pos = (item.flag or "").strip()
            if not _looks_valid_zh(word):
                continue
            base.append((word, word, pos))

        norms = [x[0] for x in base]
        out: List[str] = []
        i = 0
        while i < len(base):
            matched = self._match_longest(norms, self.zh_term_index, i)
            if matched is not None:
                span, term = matched
                # Keep matched terminology only when the full span is adjective-like.
                span_pos = [base[j][2] for j in range(i, i + span)]
                if all(pos and pos[0] in ZH_KEEP_POS_HEAD for pos in span_pos) and (not self._is_blacklisted(term)):
                    out.append(term)
                i += span
                continue
            _, keep, pos = base[i]
            if pos and pos[0] in ZH_KEEP_POS_HEAD and (not self._is_blacklisted(keep)):
                out.append(keep)
            i += 1
        return out

    def _extract_en(self, text: str) -> List[str]:
        doc = self._ensure_en_nlp()(text)
        base: List[Tuple[str, str, str]] = []
        for tok in doc:
            raw = (tok.text or "").strip()
            if not _looks_valid_en(raw):
                continue
            norm = raw.lower()
            lemma = (tok.lemma_ or raw).lower()
            upos = (tok.pos_ or "").strip()
            base.append((norm, lemma, upos))

        norms = [x[0] for x in base]
        out: List[str] = []
        i = 0
        while i < len(base):
            matched = self._match_longest(norms, self.en_term_index, i)
            if matched is not None:
                span, term = matched
                # Keep matched terminology only when the full span is adjective-like.
                span_pos = [base[j][2] for j in range(i, i + span)]
                if all(pos in EN_KEEP_POS for pos in span_pos) and (not self._is_blacklisted(term)):
                    out.append(term)
                i += span
                continue
            _, keep, upos = base[i]
            if upos in EN_KEEP_POS and (not self._is_blacklisted(keep)):
                out.append(keep)
            i += 1
        return out


def _model_label(rec: dict, path: Path) -> str:
    model = str(rec.get("model") or "").strip()
    if model:
        return model
    key = str(rec.get("model_key") or "").strip()
    if key:
        return key
    parts = path.parts
    if "results_midstream" in parts:
        idx = parts.index("results_midstream")
        if idx + 2 < len(parts):
            return f"{parts[idx+1]}/{parts[idx+2]}"
    if "results_downstream" in parts:
        idx = parts.index("results_downstream")
        if idx + 2 < len(parts):
            return f"{parts[idx+1]}/{parts[idx+2]}"
    return "unknown"


def _gender_from_downstream(rec: dict, path: Path) -> Optional[str]:
    person = str(rec.get("person") or "").strip().lower()
    if person == "xiaogang":
        return "male"
    if person == "xiaoting":
        return "female"
    name = str(rec.get("person_name") or "").strip().lower()
    if name in {"bob", "xiaogang", "小刚"}:
        return "male"
    if name in {"mary", "xiaoting", "小婷"}:
        return "female"
    folder = path.parent.name.lower()
    if folder == "xiaogang":
        return "male"
    if folder == "xiaoting":
        return "female"
    return None


def _extract_midstream_profiles(rec: dict) -> Tuple[str, str]:
    parsed = rec.get("parsed")
    if isinstance(parsed, dict):
        male = str(parsed.get("xiaogang_profile") or "").strip()
        female = str(parsed.get("xiaoting_profile") or "").strip()
        if male or female:
            return male, female

    response = str(rec.get("response") or "")
    lang = str(rec.get("language") or "zh").strip().lower()
    if not response.strip():
        return "", ""

    if lang == "en":
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


def _iter_records(root: Path, desc: str) -> Iterable[Tuple[Path, dict]]:
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


def collect_counts(
    task: str,
    root: Path,
    extractor: TokenExtractor,
    language_filter: Optional[set[str]],
    model_filters: Optional[List[str]],
) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    buckets: Dict[Tuple[str, str, str], Dict[str, object]] = defaultdict(
        lambda: {
            "male_counter": Counter(),
            "female_counter": Counter(),
            "male_total": 0,
            "female_total": 0,
            "male_docs": 0,
            "female_docs": 0,
        }
    )

    for path, rec in _iter_records(root, desc=f"scan {task}"):
        language = str(rec.get("language") or "").strip().lower()
        if language not in {"zh", "en"}:
            continue
        if language_filter and language not in language_filter:
            continue

        model = _model_label(rec, path)
        if model_filters and not any(x in model.lower() for x in model_filters):
            continue

        keys = [(task, language, model), (task, language, OVERALL_MODEL_LABEL)]

        if task == "midstream":
            male_text, female_text = _extract_midstream_profiles(rec)
            for key in keys:
                slot = buckets[key]
                if male_text:
                    male_tokens = extractor.extract(male_text, language)
                    if male_tokens:
                        slot["male_counter"].update(male_tokens)
                        slot["male_total"] += len(male_tokens)
                        slot["male_docs"] += 1
                if female_text:
                    female_tokens = extractor.extract(female_text, language)
                    if female_tokens:
                        slot["female_counter"].update(female_tokens)
                        slot["female_total"] += len(female_tokens)
                        slot["female_docs"] += 1
            continue

        text = str(rec.get("response") or "")
        if not text.strip():
            continue
        gender = _gender_from_downstream(rec, path)
        if gender is None:
            continue
        tokens = extractor.extract(text, language)
        if not tokens:
            continue
        for key in keys:
            slot = buckets[key]
            if gender == "male":
                slot["male_counter"].update(tokens)
                slot["male_total"] += len(tokens)
                slot["male_docs"] += 1
            else:
                slot["female_counter"].update(tokens)
                slot["female_total"] += len(tokens)
                slot["female_docs"] += 1

    return buckets


def build_log_or_rows(
    buckets: Dict[Tuple[str, str, str], Dict[str, object]],
    alpha: float,
    min_count: int,
    neutral_threshold: float,
    extractor: TokenExtractor,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    bucket_items = sorted(buckets.items())
    for (task, language, model), stats in tqdm(bucket_items, desc="compute log-or", unit="group"):
        male_counter: Counter = stats["male_counter"]  # type: ignore[assignment]
        female_counter: Counter = stats["female_counter"]  # type: ignore[assignment]
        total_m = int(stats["male_total"])
        total_f = int(stats["female_total"])

        vocab = set(male_counter) | set(female_counter)
        if not vocab or total_m <= 0 or total_f <= 0:
            continue
        vocab_size = len(vocab)

        term_set = extractor.zh_term_set if language == "zh" else extractor.en_term_set

        for token in vocab:
            c_m = int(male_counter.get(token, 0))
            c_f = int(female_counter.get(token, 0))
            if c_m + c_f < min_count:
                continue

            p_m = (c_m + alpha) / (total_m + alpha * vocab_size)
            p_f = (c_f + alpha) / (total_f + alpha * vocab_size)
            log_or = math.log(p_m / p_f)

            if log_or > neutral_threshold:
                label = "male_biased"
            elif log_or < -neutral_threshold:
                label = "female_biased"
            else:
                label = "neutral"

            rows.append(
                {
                    "task": task,
                    "language": language,
                    "model": model,
                    "token": token,
                    "count_m": c_m,
                    "count_f": c_f,
                    "total_m": total_m,
                    "total_f": total_f,
                    "vocab_size": vocab_size,
                    "alpha": alpha,
                    "p_m": p_m,
                    "p_f": p_f,
                    "log_or": log_or,
                    "label": label,
                    "is_term": token in term_set,
                    "male_docs": int(stats["male_docs"]),
                    "female_docs": int(stats["female_docs"]),
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_topk_rows(rows: List[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["language"]), str(row["model"]))].append(row)

    out: List[Dict[str, object]] = []
    for (task, language, model), items in sorted(grouped.items()):
        sorted_items = sorted(items, key=lambda x: float(x["log_or"]), reverse=True)
        top_m = sorted_items[:top_k]
        top_f = sorted(items, key=lambda x: float(x["log_or"]))[:top_k]
        for rank, row in enumerate(top_m, start=1):
            out.append(
                {
                    "task": task,
                    "language": language,
                    "model": model,
                    "side": "male_top",
                    "rank": rank,
                    "token": row["token"],
                    "log_or": row["log_or"],
                    "count_m": row["count_m"],
                    "count_f": row["count_f"],
                    "is_term": row["is_term"],
                }
            )
        for rank, row in enumerate(top_f, start=1):
            out.append(
                {
                    "task": task,
                    "language": language,
                    "model": model,
                    "side": "female_top",
                    "rank": rank,
                    "token": row["token"],
                    "log_or": row["log_or"],
                    "count_m": row["count_m"],
                    "count_f": row["count_f"],
                    "is_term": row["is_term"],
                }
            )
    return out


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # type: ignore[return-value]
    for row in rows:
        key = (str(row["task"]), str(row["language"]), str(row["model"]))
        label = str(row["label"])
        grouped[key][label] += 1
        grouped[key]["all"] += 1

    payload = []
    for (task, language, model), counters in sorted(grouped.items()):
        payload.append(
            {
                "task": task,
                "language": language,
                "model": model,
                "token_rows": counters.get("all", 0),
                "male_biased": counters.get("male_biased", 0),
                "female_biased": counters.get("female_biased", 0),
                "neutral": counters.get("neutral", 0),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    language_filter = None if args.language == "all" else {args.language}
    model_filters = [x.lower() for x in (args.model or [])]

    print("[start] Loading term lexicons...", flush=True)
    zh_terms = _load_terms(args.zh_terms, "zh")
    en_terms = _load_terms(args.en_terms, "en")
    blacklist = _load_blacklist(args.blacklist)
    extractor = TokenExtractor(zh_terms=zh_terms, en_terms=en_terms, blacklist=blacklist)
    print(
        (
            f"[start] Terms loaded: zh={len(zh_terms)}, en={len(en_terms)}, "
            f"blacklist={len(blacklist)}; task={args.task}, language={args.language}"
        ),
        flush=True,
    )

    all_buckets: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    if args.task in {"midstream", "both"}:
        mid = collect_counts(
            task="midstream",
            root=args.midstream_root,
            extractor=extractor,
            language_filter=language_filter,
            model_filters=model_filters,
        )
        all_buckets.update(mid)
    if args.task in {"downstream", "both"}:
        down = collect_counts(
            task="downstream",
            root=args.downstream_root,
            extractor=extractor,
            language_filter=language_filter,
            model_filters=model_filters,
        )
        all_buckets.update(down)

    rows = build_log_or_rows(
        buckets=all_buckets,
        alpha=args.alpha,
        min_count=args.min_count,
        neutral_threshold=args.neutral_threshold,
        extractor=extractor,
    )
    topk_rows = build_topk_rows(rows, top_k=args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "logor_results.csv"
    topk_path = args.output_dir / "logor_topk.csv"
    summary_path = args.output_dir / "logor_summary.json"

    detail_fields = [
        "task",
        "language",
        "model",
        "token",
        "count_m",
        "count_f",
        "total_m",
        "total_f",
        "vocab_size",
        "alpha",
        "p_m",
        "p_f",
        "log_or",
        "label",
        "is_term",
        "male_docs",
        "female_docs",
    ]
    topk_fields = [
        "task",
        "language",
        "model",
        "side",
        "rank",
        "token",
        "log_or",
        "count_m",
        "count_f",
        "is_term",
    ]

    write_csv(detail_path, rows, detail_fields)
    write_csv(topk_path, topk_rows, topk_fields)
    write_summary(summary_path, rows)

    print(f"Wrote: {detail_path.as_posix()} (rows={len(rows)})")
    print(f"Wrote: {topk_path.as_posix()} (rows={len(topk_rows)})")
    print(f"Wrote: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
