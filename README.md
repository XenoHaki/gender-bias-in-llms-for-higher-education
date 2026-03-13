# AI Education Bias Runner

This project automates the “上游” experiments described in `docs/prompt/prompt.docx`: it prepares and fires the STEM
subject and ability vocabulary prompts against multiple Chinese and English LLM providers, captures the
raw responses, and stores them for downstream analysis.

## Project structure

- `subjects_task.py` / `abilities_task.py` – entry points for the two required prompt suites.
- `utils.py` – shared helpers for loading configs, generating prompts, randomising vocab order, and
  writing JSONL outputs.
- `prompts/` – dataset definitions and language-specific prompt templates.
- `clients/` – API adapters for ChatGPT, Gemini, Claude, Llama (Groq), DeepSeek, Doubao, ERNIE,
  Qwen, and Kimi.
- `config/models.yaml` – central configuration for default run parameters and model metadata.
- `data/results/` – per-model JSONL logs (gitignored) containing prompts, metadata, and responses.

## Requirements & installation

Python 3.10+ is recommended. After creating/activating a virtual environment, install dependencies:

```bash
pip install -e .
```

`pyproject.toml` also defines optional dev tooling (Ruff) if you need linting.

## Configuring providers & versions

`config/models.yaml` defines defaults (temperature, token limits, etc.) and lists each
model family. Every family can hold multiple `versions`, so you can test, for example, GPT-4o mini and a
future GPT-4o release independently:

```yaml
- name: chatgpt
  display_name: ChatGPT
  provider: openai
  env: OPENAI_API_KEY
  base_url: https://api.openai.com/v1/chat/completions
  versions:
    - key: gpt4o-mini
      display_name: GPT-4o mini
      model_id: gpt-4o-mini
    # - key: gpt4o
    #   display_name: GPT-4o
    #   model_id: gpt-4o
```

Edit this file to change model IDs, add new releases, or disable providers. The current entries expect
these environment variables before running a task:

| Provider / Model | Environment variables |
| ---------------- | --------------------- |
| ChatGPT (OpenAI) | `OPENAI_API_KEY` |
| Gemini           | `GEMINI_API_KEY` |
| Claude           | `ANTHROPIC_API_KEY` |
| Llama (Groq)     | `GROQ_API_KEY` |
| DeepSeek-V3.1 (SiliconFlow) | `SILICONFLOW_API_KEY` |
| Doubao           | `DOUBAO_API_KEY` |
| ERNIE-4.5-300B-A47B (SiliconFlow) | `SILICONFLOW_API_KEY` |
| Qwen3-32B (SiliconFlow) | `SILICONFLOW_API_KEY` |
| Kimi-K2-Instruct-0905 (SiliconFlow) | `SILICONFLOW_API_KEY` |

You can comment out entries you do not plan to run, use `--model` to target a single family, or use
`--model-version` to focus on a specific release (the YAML `key` field).

> **SiliconFlow tip**  
> 硅基流动提供了完全兼容 OpenAI Chat Completions 的接口。DeepSeek-V3.1、Kimi-K2-Instruct-0905、
> Qwen3-32B 和 ERNIE-4.5-300B-A47B 都使用 `https://api.siliconflow.cn/v1/chat/completions`，并通过
> `Authorization: Bearer <token>`（`<token>` 就是 `SILICONFLOW_API_KEY`）鉴权。官方示例
> `requests.post(..., json={"model": "...", "messages": [...]})` 与本项目客户端完全一致；如果要添加
> 新的硅基流动模型，只需在 `config/models.yaml` 中新增版本并填入对应 `model_id`。

## Running the tasks

Both entry points use `typer`, so `--help` shows all switches.

```bash
# Dry-run (no API calls) to inspect prompts
python subjects_task.py --dry-run --debug

# Execute the subject vocabulary study for ChatGPT & Gemini only
python subjects_task.py --model chatgpt --model gemini --seed 42

# Run only the GPT-4o mini variant with verbose logs
python subjects_task.py --model chatgpt --model-version gpt4o-mini --debug

# Execute the ability vocabulary study (all models, zh + en)
python abilities_task.py
```

Key behaviours:

- Every combination of model × language × prompt variant executes 10 runs
  (5 prompts with the first name order, 5 with the reversed order) as required.
- Word order inside each prompt is randomized on every run.
- `--seed` fixes the random order to reproduce a batch.
- `--language zh` or `--language en` restricts the calls; omit the flag to run both.
- `--model` accepts one or more model families (chatgpt, gemini, claude, etc.); omit for all models.
- `--model-version` narrows the run to version keys declared in `config/models.yaml`.
- `--workers` controls how many prompts run concurrently (default 1); increase it to speed up large batches.
- `--debug` turns on verbose logging and prints raw responses.
- `--dry-run` prints prompt previews without touching any API.
- A tqdm progress bar tracks how many prompts have been executed.
- Each prompt send automatically retries up to `max_attempts` times (configurable in `defaults`) with exponential backoff before surfacing an error.

Results overwrite any previous run and are appended to
`data/results/<task>/<model-family>/<version>/<language>/<variant>/standard.jsonl`
with rich metadata (prompt preview, shuffled vocabulary, name order, errors, etc.), making it easy to
aggregate downstream.

Every run truncates the previous JSONL files in the same directory before writing new entries, so each execution keeps only its own data. Logs stream both to the console and `logs/runner.log`, so if a run aborts early you can open that file to inspect the traceback without rerunning with manual redirection.

## Midstream scenarios

`midstream_task.py` implements 8 narrative scenarios（大学报考、大学选课、学业表现、课堂展示、学术竞赛、科研项目、毕业设计、保研申请）。每个场景包含若干 “选项组”（例如表格中的 A/B、C/D、E/E 理论），代码会对每个模型 × 场景 × 选项组执行 10 次对话（5 次先描述小刚，5 次先描述小婷），并要求模型以 JSON 形式返回两人的简介与分配结果。

```bash
# 运行所有场景
python midstream_task.py --model deepseek --model-version deepseek-v31 --workers 2 --seed 42

# 仅运行学术竞赛场景（s5_competition）
python midstream_task.py --model qwen --model-version qwen3-32b --scenario s5_competition --debug
```

结果会存放在 `data/results_midstream/<model>/<version>/<scenario>/<combo>/standard.jsonl`，每条记录包含原始 prompt、模型响应、解析后的 JSON（若成功）以及潜在的解析错误，方便后续统计。

## Next steps

- Implement the “中游”和“下游” analyses referenced in the prompt onto the stored JSONL data.
- Add evaluation notebooks/dashboards to quantify bias signals.
- Introduce automated retries/backoff tailored per provider or integrate streaming so responses are
  flushed in real time.
- Wire up unit tests or mocks for each client adapter to guard against breaking API changes.
