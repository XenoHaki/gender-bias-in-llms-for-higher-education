from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE_DIR = Path("data/results")
OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MALE_NAMES = {"小刚", "Bob"}
FEMALE_NAMES = {"小婷", "Mary"}
NAME_LABELS = {"male": "小刚 / Bob", "female": "小婷 / Mary"}


def configure_fonts():
    # Try to use a Chinese-capable font if available, fall back to default.
    preferred = ["SimHei", "Microsoft YaHei", "Microsoft JhengHei", "PingFang SC", "Hiragino Sans GB"]
    found = None
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            found = name
            break
    if found:
        plt.rcParams["font.sans-serif"] = [found]
    plt.rcParams["axes.unicode_minus"] = False


def canonical_name(name: str) -> str | None:
    if name in MALE_NAMES:
        return "male"
    if name in FEMALE_NAMES:
        return "female"
    return None


def parse_matching_line(line: str) -> Tuple[str, str] | None:
    if "-" not in line:
        return None
    word, name = line.rsplit("-", 1)
    word = word.strip()
    name = name.strip()
    if not word or not name:
        return None
    return word, name


def parse_probability_line(line: str) -> Tuple[str, float, float] | None:
    try:
        word, first, second = line.rsplit("-", 2)
    except ValueError:
        return None
    word = word.strip()
    first = first.strip().rstrip("%")
    second = second.strip().rstrip("%")
    if not word:
        return None
    try:
        first_val = float(first) / 100 if "%" in line else float(first)
        second_val = float(second) / 100 if "%" in line else float(second)
    except ValueError:
        return None
    return word, first_val, second_val


def collect_data():
    matching_counts = defaultdict(lambda: defaultdict(lambda: {"male": 0, "female": 0}))
    probability_stats = defaultdict(
        lambda: defaultdict(
            lambda: {"male": {"sum": 0.0, "count": 0}, "female": {"sum": 0.0, "count": 0}}
        )
    )

    for jsonl_path in BASE_DIR.rglob("standard.jsonl"):
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                task = record.get("task")
                variant = record.get("variant")
                response = record.get("response")
                if not response:
                    continue

                if variant == "matching":
                    for raw_line in response.splitlines():
                        parsed = parse_matching_line(raw_line.strip())
                        if not parsed:
                            continue
                        word, assigned_name = parsed
                        canonical = canonical_name(assigned_name)
                        if not canonical:
                            continue
                        matching_counts[task][word][canonical] += 1
                elif variant == "probability":
                    name_order = record.get("name_order") or []
                    for raw_line in response.splitlines():
                        parsed = parse_probability_line(raw_line.strip())
                        if not parsed or len(name_order) < 2:
                            continue
                        word, first_prob, second_prob = parsed
                        for name, prob in zip(name_order, (first_prob, second_prob)):
                            canonical = canonical_name(name)
                            if not canonical:
                                continue
                            stats = probability_stats[task][word][canonical]
                            stats["sum"] += prob
                            stats["count"] += 1
    return matching_counts, probability_stats


def write_tables(matching_counts, probability_stats):
    matching_path = OUTPUT_DIR / "matching_counts.csv"
    probability_path = OUTPUT_DIR / "probability_means.csv"

    with matching_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task", "word", "male_count", "female_count", "total"])
        for task, words in sorted(matching_counts.items()):
            for word, counts in sorted(words.items()):
                male = counts["male"]
                female = counts["female"]
                writer.writerow([task, word, male, female, male + female])

    with probability_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task", "word", "male_mean", "female_mean"])
        for task, words in sorted(probability_stats.items()):
            for word, stats in sorted(words.items()):
                male_mean = (
                    stats["male"]["sum"] / stats["male"]["count"] if stats["male"]["count"] else 0.0
                )
                female_mean = (
                    stats["female"]["sum"] / stats["female"]["count"]
                    if stats["female"]["count"]
                    else 0.0
                )
                writer.writerow([task, word, f"{male_mean:.4f}", f"{female_mean:.4f}"])


def plot_matching(matching_counts):
    for task, words in matching_counts.items():
        ordered_words = sorted(words.keys())
        male = [words[w]["male"] for w in ordered_words]
        female = [words[w]["female"] for w in ordered_words]

        fig, ax = plt.subplots(figsize=(12, 0.4 * len(ordered_words) + 2))
        bars_male = ax.barh(ordered_words, male, label=NAME_LABELS["male"])
        bars_female = ax.barh(ordered_words, female, left=male, label=NAME_LABELS["female"])
        ax.bar_label(bars_male, labels=[str(v) for v in male], label_type="center", color="white")
        ax.bar_label(
            bars_female,
            labels=[str(v) for v in female],
            label_type="center",
            color="white",
        )
        ax.set_xlabel("匹配次数")
        ax.set_title(f"{task} 任务：匹配次数分布")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{task}_matching_counts.png", dpi=200)
        plt.close(fig)


def plot_probability(probability_stats):
    for task, words in probability_stats.items():
        ordered_words = sorted(words.keys())
        male_means = [
            words[w]["male"]["sum"] / words[w]["male"]["count"]
            if words[w]["male"]["count"]
            else 0.0
            for w in ordered_words
        ]
        female_means = [
            words[w]["female"]["sum"] / words[w]["female"]["count"]
            if words[w]["female"]["count"]
            else 0.0
            for w in ordered_words
        ]

        y_pos = list(range(len(ordered_words)))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 0.4 * len(ordered_words) + 2))
        male_bars = ax.barh(
            [i + width / 2 for i in y_pos],
            male_means,
            height=width,
            label=NAME_LABELS["male"],
        )
        female_bars = ax.barh(
            [i - width / 2 for i in y_pos],
            female_means,
            height=width,
            label=NAME_LABELS["female"],
        )
        ax.bar_label(male_bars, labels=[f"{v:.2f}" for v in male_means], label_type="center")
        ax.bar_label(female_bars, labels=[f"{v:.2f}" for v in female_means], label_type="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ordered_words)
        ax.set_xlabel("平均概率")
        ax.set_title(f"{task} 任务：概率均值分布")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{task}_probability_means.png", dpi=200)
        plt.close(fig)


def main():
    configure_fonts()
    matching_counts, probability_stats = collect_data()
    write_tables(matching_counts, probability_stats)
    plot_matching(matching_counts)
    plot_probability(probability_stats)
    print("结果写入 analysis/ 目录，包括 CSV 和 PNG 图表。")


if __name__ == "__main__":
    main()
