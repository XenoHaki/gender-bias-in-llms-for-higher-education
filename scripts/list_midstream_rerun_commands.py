from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from utils import load_model_settings


DEFAULT_REPORT = Path("analysis/mixed_language_report.json")
ALL_SCENARIOS = (
    "s1_major",
    "s2_course",
    "s3_performance",
    "s4_presentation",
    "s5_competition",
    "s6_research",
    "s7_thesis",
    "s8_grad_application",
)


def _load_scopes(report_path: Path) -> Dict[Tuple[str, str, str], Set[str]]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    raw_scopes: Sequence[Sequence[str]] = data.get("rerun_scopes", {}).get("midstream", [])
    grouped: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    for scope in raw_scopes:
        if len(scope) != 4:
            continue
        family, version, language, scenario = scope
        grouped[(str(family), str(version), str(language))].add(str(scenario))
    return grouped


def _configured_model_keys() -> Set[Tuple[str, str]]:
    _, models = load_model_settings()
    return {
        ((m.family or m.name).lower(), (m.version or "default").lower())
        for m in models
    }


def _cmd_for_scope(
    family: str,
    version: str,
    language: str,
    scenarios: Iterable[str],
    workers: int | None,
) -> str:
    base = (
        f"python midstream_task.py --model {family} --model-version {version} --language {language}"
    )
    scenario_set = set(scenarios)
    if scenario_set and scenario_set != set(ALL_SCENARIOS):
        for scenario_id in sorted(scenario_set):
            base += f" --scenario {scenario_id}"
    if workers is not None:
        base += f" --workers {workers}"
    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print targeted midstream rerun commands from mixed-language report."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Path to report JSON (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional workers value to append to commands.",
    )
    args = parser.parse_args()

    if not args.report.exists():
        raise FileNotFoundError(f"Missing report file: {args.report}")

    grouped = _load_scopes(args.report)
    configured = _configured_model_keys()

    runnable: List[str] = []
    missing: List[str] = []

    for (family, version, language), scenarios in sorted(grouped.items()):
        cmd = _cmd_for_scope(
            family=family,
            version=version,
            language=language,
            scenarios=scenarios,
            workers=args.workers,
        )
        if (family.lower(), version.lower()) in configured:
            runnable.append(cmd)
        else:
            missing.append(cmd)

    print("# Runnable now (present in config/models.yaml)")
    for cmd in runnable:
        print(cmd)
    if not runnable:
        print("(none)")

    print("\n# Missing from config/models.yaml (legacy model keys)")
    for cmd in missing:
        print(cmd)
    if not missing:
        print("(none)")


if __name__ == "__main__":
    main()
