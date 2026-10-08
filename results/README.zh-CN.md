# results/ — v5 实验输出（15 模型 × 15 个当前场景）

*[English](README.md) | 中文*

本目录为主实验输出，对应 `config.yaml` 的 `experiment.output_dir: ./results`。逐单元实验数据仅保留当前启用的场景；历史汇总文件仍保留 30 个场景的统计行，便于复核整体统计口径。

## 内容

- **`conditions_results.json`** —— 当前场景的合并评分表：15 模型 × 15 场景 × 3 变体 × 4 条件
- **`run_YYYYMMDD_HHMMSS/`** —— 15 个原始 run 目录（每个 run = 1 个模型的当前场景输出），包含：
  - `conditions_results.json`（该模型的当前场景评分）
  - `gains_summary.json` / `gains_summary.csv`（该模型的 paired gains ）
  - `baselines.json`（随机基线）
  - `outputs.jsonl`（当前场景的原始 LLM 输出）
  - `units/`（当前场景每 unit 的 manifest / trajectory / final / evidence / snapshots / prompts）
  - `run_manifest.json`、`experiment.log`

## 模型与 run 对应关系

| 模型 | run 目录 |
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
