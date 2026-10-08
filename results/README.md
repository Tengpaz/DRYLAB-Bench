# results/ — v5 experiment outputs (15 models × 15 current scenarios)

*English | [中文](README.zh-CN.md)*

This directory holds the main experiment outputs and corresponds to `experiment.output_dir: ./results` in `config.yaml`. Per-unit experiment data is kept only for the currently enabled scenarios; historical summary files still retain statistics rows for 30 scenarios so the overall scoring conventions can be reviewed.

## Contents

- **`conditions_results.json`** — merged score table for the current scenarios: 15 models × 15 scenarios × 3 variants × 4 conditions
- **`run_YYYYMMDD_HHMMSS/`** — 15 raw run directories (each run = one model's outputs for the current scenarios), containing:
  - `conditions_results.json` (that model's scores for the current scenarios)
  - `gains_summary.json` / `gains_summary.csv` (that model's paired gains)
  - `baselines.json` (random baselines)
  - `outputs.jsonl` (raw LLM outputs for the current scenarios)
  - `units/` (per-unit manifest / trajectory / final / evidence / snapshots / prompts for the current scenarios)
  - `run_manifest.json`, `experiment.log`

## Model-to-run mapping

| Model | Run directory |
|---|---|
| kimi-k3 | run_20260907_123243 |
| claude-opus-5 | run_20260907_231110 |
| claude-opus-4-8 | run_20260908_005612 |
| gpt-5.6-luna | run_20260908_020812 |
| gpt-5.6-sol | run_20260908_040514 |
| gpt-5.6-terra | run_20260908_065146 |
| gemini-3.6-flash | run_20260908_092448 |
| qwen3.8-max | run_20260908_130350 |
| deepseek-v4-flash | run_20260909_031730 |
| grok-4.5 | run_20260909_142239 |
| deepseek-v4-pro | run_20260909_213842 |
| glm-5.3 | run_20260910_114404 |
| gemini-3.8-flash | run_20260910_165902 |
| gpt-6-astra | run_20260911_033107 |
| claude-fable-5 | run_20260911_103843 |
