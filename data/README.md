# data/ — Re-acquiring large datasets

*English | [中文](README.zh-CN.md)*

This directory only retains the versioned curated small file `h3_to_h5_map.json`. The ground-truth data, tools, and weights required to run experiments must be re-obtained or recomputed from **public sources** as described below.

## 1. ProteinGym (primary GT data, ~2 GB)

```bash
python -m drylab_bench.download_datasets --data-dir ./data
```

Downloads the 5 parquet shards of `OATML-Markslab/ProteinGym_v1` from Hugging Face and merges them into `data/ProteinGym/DMS_substitutions.csv`.

## 2. ViroGym (primary GT data, ~1.1 GB)

```bash
git clone https://github.com/GSK-AI/viroGym data/ViroGym
```

Uses `DMS/benchmark.csv` + `DMS/cleaned_benchmark/`. **Never move or restructure anything under it.**

## 3. Data index

```bash
python -m drylab_bench.data_prep --data-dir ./data   # validate + generate data/data_index.json
```

## 4. BT evidence (`data/bt_evidence/`, ~5 GB)

- **Computed live at runtime, no precomputed tables needed:** `sequence_plm` (ESM-1v, `cache_file: null` in `config.yaml`), `evolution_msa` (MMseqs2 → `sprotDB`), and FoldX PositionScan for `phenotype` (live).
- **Requires pre-download or pre-computation:** per-task `structure_scores.json` (ProteinMPNN), `phenotype.json` (AlphaMissense / EVEscape), `<acc>.gff` (UniProt), `sprotDB*` (Swiss-Prot search database), experimental PDBs (RCSB PDB), `wt_sequences.json`, and FoldX `energies_*.txt`.
- **External tool binaries:** MMseqs2 (`./.tools/mmseqs`), FoldX 5.1 (`./.tools/foldx`), ProteinMPNN (`./.tools/ProteinMPNN`) — obtain these from their respective official distribution channels.

## 5. In-silico weights/data (`data/insilico/`, ~2.5 GB)

- **ESM-1v seeds 1–5:** Hugging Face `facebook/esm1v_t33_650M_UR90S_{1..5}` → `data/insilico/esm1v_models/` (corresponds to `sequence_plm.model_root` in `config.yaml`, `n_seeds: 5`).
- **ESM-2:** Hugging Face `facebook/esm2_t33_650M_UR50D` (used by the evaluator, served from the HF cache, ~4.9 GB).
- **EVE TP53:** `evemodel.org` API → `data/insilico/EVE_TP53_predictions.csv`.
- **AlphaMissense:** Zenodo `8208688` / Google `dm_alphamissense` (extract tables on demand only when generating `phenotype.json`).
- **ToxinPred3:** models bundled with the `toxinpred3` PyPI package, exported as `toxinpred3_trees.npz`.
