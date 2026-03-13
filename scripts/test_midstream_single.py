from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from midstream_task import SCENARIO_LOOKUP, build_prompt_text, localize_option
from utils import load_model_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single midstream debug run: print prompt, request params, and model output."
    )
    parser.add_argument("--model", default="gemini", help="Model family, e.g. gemini")
    parser.add_argument(
        "--model-version",
        default="gemini-3-pro-preview",
        help="Model version key, e.g. gemini-3-pro-preview",
    )
    parser.add_argument("--scenario", default="s3_performance", help="Scenario id, e.g. s3_performance")
    parser.add_argument("--combo-index", type=int, default=1, help="1-based combo index in scenario")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Prompt language")
    parser.add_argument(
        "--first",
        choices=["xiaogang", "xiaoting"],
        default="xiaogang",
        help="Who is described first in prompt",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max output tokens")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")
    parser.add_argument(
        "--raw-response-path",
        default="analysis/midstream_single_raw_response.json",
        help="Path to save full raw JSON response",
    )
    return parser.parse_args()


def _mask_secret(value: str) -> str:
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]}"


def _pick_model(model_family: str, model_version: str):
    _, models = load_model_settings()
    family_l = model_family.lower()
    version_l = model_version.lower()
    for model in models:
        fam = (model.family or model.name).lower()
        ver = (model.version or "default").lower()
        if fam == family_l and ver == version_l:
            return model
    raise ValueError(f"Model not found: {model_family}/{model_version}")


def _name_order(language: str, first: str) -> Tuple[str, str]:
    if language == "zh":
        xiaogang, xiaoting = "小刚", "小婷"
    else:
        xiaogang, xiaoting = "Bob", "Mary"
    if first == "xiaoting":
        return xiaoting, xiaogang
    return xiaogang, xiaoting


def _build_prompt(args: argparse.Namespace) -> Tuple[str, str, str]:
    scenario = SCENARIO_LOOKUP.get(args.scenario)
    if scenario is None:
        raise ValueError(f"Unknown scenario id: {args.scenario}")
    if args.combo_index < 1 or args.combo_index > len(scenario.option_pairs):
        raise ValueError(f"combo-index out of range: 1..{len(scenario.option_pairs)}")

    option_a_raw, option_b_raw = scenario.option_pairs[args.combo_index - 1]
    option_a = localize_option(option_a_raw, args.language)
    option_b = localize_option(option_b_raw, args.language)
    prompt = build_prompt_text(
        scenario=scenario,
        option_a=option_a,
        option_b=option_b,
        name_order=_name_order(args.language, args.first),
        language=args.language,
    )
    return prompt, option_a, option_b


def _build_request(
    provider: str,
    model_id: str,
    base_url: str | None,
    api_key: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    provider_l = provider.lower()
    if provider_l == "gemini":
        base = base_url.rstrip("/") if base_url else "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{model_id}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        return url, headers, payload

    # OpenAI-compatible route used by current n1n.ai Gemini config.
    url = base_url or "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a precise assistant for bias diagnostics."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return url, headers, payload


def _extract_output(provider: str, data: Dict[str, Any]) -> str:
    provider_l = provider.lower()
    if provider_l == "gemini":
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts).strip()
    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    args = parse_args()
    load_dotenv()

    defaults, _ = load_model_settings()
    model = _pick_model(args.model, args.model_version)
    prompt, option_a, option_b = _build_prompt(args)

    temperature = float(args.temperature if args.temperature is not None else defaults.get("temperature", 0.8))
    max_tokens = int(args.max_tokens if args.max_tokens is not None else defaults.get("max_output_tokens", 800))

    api_key = os.getenv(model.env)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {model.env}")

    url, headers, payload = _build_request(
        provider=model.provider,
        model_id=model.model_id,
        base_url=model.base_url,
        api_key=api_key,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    request_debug = {
        "model_family": model.family,
        "model_version": model.version,
        "model_id": model.model_id,
        "provider": model.provider,
        "scenario": args.scenario,
        "combo_index": args.combo_index,
        "option_a": option_a,
        "option_b": option_b,
        "language": args.language,
        "url": url,
        "headers": {
            k: (_mask_secret(v) if k.lower() == "authorization" else v) for k, v in headers.items()
        },
        "payload": payload,
        "timeout_seconds": args.timeout,
    }

    print("=== PROMPT ===")
    print(prompt)
    print("\n=== API PARAMS (SENT) ===")
    print(json.dumps(request_debug, ensure_ascii=False, indent=2))

    with httpx.Client(timeout=args.timeout) as client:
        response = client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    model_output = _extract_output(model.provider, data)

    raw_path = Path(args.raw_response_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== MODEL OUTPUT ===")
    print(model_output)
    print("\n=== RESPONSE META ===")
    print(
        json.dumps(
            {
                "status_code": response.status_code,
                "raw_response_saved_to": str(raw_path).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
