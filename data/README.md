# data/ — 大体积数据重新获取说明

本目录只保留版本化的 curated 小文件 `h3_to_h5_map.json`。运行实验所需的 GT 数据与工具/权重，请按下述方式从**公开来源**重新获取/重算。

## 1. ProteinGym（GT 主数据，~2 GB）

```bash
python -m drylab_bench.download_datasets --data-dir ./data
```

从 Hugging Face `OATML-Markslab/ProteinGym_v1` 下载 5 个 parquet 分片并合并为 `data/ProteinGym/DMS_substitutions.csv`。

## 2. ViroGym（GT 主数据，~1.1 GB）

```bash
git clone https://github.com/GSK-AI/viroGym data/ViroGym
```

使用 `DMS/benchmark.csv` + `DMS/cleaned_benchmark/`。**永不移动/重构其下任何东西**。

## 3. 数据索引

```bash
python -m drylab_bench.data_prep --data-dir ./data   # 校验 + 生成 data/data_index.json
```

## 4. BT evidence（`data/bt_evidence/`，~5 GB）

- **运行时直接计算、无需预计算表**：`sequence_plm`（ESM-1v，`config.yaml` 中 `cache_file: null`）、`evolution_msa`（MMseqs2 → `sprotDB`）、`phenotype` 的 FoldX PositionScan（live）。
- **需要预下载/预计算**：per-task `structure_scores.json`（ProteinMPNN）、`phenotype.json`（AlphaMissense / EVEscape）、`<acc>.gff`（UniProt）、`sprotDB*`（Swiss-Prot 检索库）、实验 PDB（RCSB PDB）、`wt_sequences.json`、FoldX 的 `energies_*.txt`。
- **外部工具二进制**：MMseqs2（`./.tools/mmseqs`）、FoldX 5.1（`./.tools/foldx`）、ProteinMPNN（`./.tools/ProteinMPNN`）—— 从各自官方发行渠道获取。

## 5. in silico 权重/数据（`data/insilico/`，~2.5 GB）

- **ESM-1v seeds 1–5**：Hugging Face `facebook/esm1v_t33_650M_UR90S_{1..5}` → `data/insilico/esm1v_models/`（对应 `config.yaml` 的 `sequence_plm.model_root`，`n_seeds: 5`）。
- **ESM-2**：Hugging Face `facebook/esm2_t33_650M_UR50D`（evaluator 用，走 HF 缓存，~4.9 GB）。
- **EVE TP53**：`evemodel.org` API → `data/insilico/EVE_TP53_predictions.csv`。
- **AlphaMissense**：Zenodo `8208688` / Google `dm_alphamissense`（仅用于生成 `phenotype.json` 时临时抽表）。
- **ToxinPred3**：`toxinpred3` PyPI 包自带模型导出为 `toxinpred3_trees.npz`。
