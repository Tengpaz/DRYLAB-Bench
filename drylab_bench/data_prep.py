#!/usr/bin/env python3
"""
DRYLAB-Bench — Data Preparation Script

Validates downloaded datasets, maps column names, and creates index files
for efficient querying during experiments.

Usage:
    python data_prep.py --data-dir ./data
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_prep")


# ============================================================================
# Column name mappings for ProteinGym v1
# ============================================================================

# ProteinGym v1 DMS_substitutions.csv actual columns (from NeurIPS 2023 paper):
# - DMS_id: assay identifier (e.g., "TP53_HUMAN_Giacomelli_NULL_2018")
# - mutant: mutation string, e.g. "A1C", "K417N"
# - mutated_sequence: full mutated amino acid sequence
# - DMS_score: normalized fitness score (float, typically centered around 0)
# - DMS_score_bin: binarized score (0/1, whether score > 0)
# - Includes both substitution and indel assays

# We need to identify specific assays for our four scenarios:
ASSAY_FILTERS = {
    "TP53_HUMAN": {
        "description": "Human TP53 tumor suppressor",
        "patterns": ["TP53", "P53", "p53"],
        "min_mutations": 200,
        "columns": {
            "fitness_score": "DMS_score",
        },
    },
    "BLAT_ECOLX": {
        "description": "TEM-1 beta-lactamase (E. coli)",
        "patterns": ["BLAT", "TEM-1", "TEM1", "bla", "beta-lactamase", "lactamase"],
        "min_mutations": 200,
        "columns": {
            # Deng 2012 = ampicillin-resistance fitness (proxy for β-lactam
            # resistance engineering, NOT carbapenem hydrolysis)
            "fitness_score": "DMS_score",
        },
    },
}


def detect_assay_ids(df: pd.DataFrame, target: dict) -> list:
    """Find DMS assay IDs matching the target protein."""
    if "DMS_id" not in df.columns:
        logger.warning("No DMS_id column found in data")
        return []

    unique_ids = df["DMS_id"].dropna().unique()
    matches = []

    for dms_id in sorted(unique_ids):
        for pattern in target["patterns"]:
            if pattern.upper() in str(dms_id).upper():
                n_mut = (df["DMS_id"] == dms_id).sum()
                if n_mut >= target["min_mutations"]:
                    matches.append((dms_id, n_mut))
                    break

    return sorted(matches, key=lambda x: -x[1])  # sort by n_mutations descending


def validate_proteingym(data_dir: Path) -> dict:
    """
    Validate ProteinGym v1 data and identify assay IDs for target proteins.

    Returns a dict with assay ID mappings for config.yaml.
    """
    logger.info("=" * 50)
    logger.info("Validating ProteinGym data...")
    logger.info("=" * 50)

    csv_path = data_dir / "ProteinGym" / "DMS_substitutions.csv"
    raw_dir = data_dir / "ProteinGym" / "_raw"

    if csv_path.exists():
        logger.info(f"Reading {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        # Try Parquet shards (ProteinGym v1 on HuggingFace is Parquet-only)
        pq_patterns = [
            raw_dir / "DMS_substitutions",
            raw_dir,
        ]
        pq_files = []
        for pat in pq_patterns:
            if pat.exists() and pat.is_dir():
                pq_files = sorted(pat.glob("*.parquet"))
            if pq_files:
                break

        if pq_files:
            logger.info(f"Reading {len(pq_files)} Parquet shard(s) from {pq_files[0].parent}...")
            dfs = [pd.read_parquet(p) for p in pq_files]
            df = pd.concat(dfs, ignore_index=True)
        elif (data_dir / "ProteinGym" / "DMS_substitutions.zip").exists():
            zip_path = data_dir / "ProteinGym" / "DMS_substitutions.zip"
            logger.info(f"Found zip file: {zip_path}. Extracting...")
            df = pd.read_csv(zip_path)
        else:
            logger.error(f"ProteinGym data not found at {csv_path} or as Parquet shards.")
            logger.info("Run: python download_datasets.py --data-dir ./data")
            return {}

    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Unique DMS assays: {df['DMS_id'].nunique() if 'DMS_id' in df.columns else 'N/A'}")

    # Check for required columns
    required = ["DMS_id", "mutant", "DMS_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.info("Expected columns in ProteinGym v1:")
        logger.info("  DMS_id, mutant, mutated_sequence, DMS_score, DMS_score_bin, target_seq")
        return {}

    # Detect assays for our target proteins
    assay_mapping = {}

    for protein_id, target in ASSAY_FILTERS.items():
        logger.info(f"\nSearching for {protein_id} ({target['description']})...")
        matches = detect_assay_ids(df, target)
        if matches:
            for dms_id, n_mut in matches[:5]:  # Show top 5 matches
                logger.info(f"  {dms_id}: {n_mut:,} mutations")
            best_id = matches[0][0]
            assay_mapping[protein_id] = best_id
            logger.info(f"  → Selected: {best_id}")
        else:
            logger.warning(f"  No matching assays found for {protein_id}")

    # Summary stats for matched assays
    for protein_id, assay_id in assay_mapping.items():
        subset = df[df["DMS_id"] == assay_id]
        scores = subset["DMS_score"].dropna()
        logger.info(f"\n{protein_id} ({assay_id}):")
        logger.info(f"  Mutations: {len(subset):,}")
        logger.info(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        logger.info(f"  Score mean: {scores.mean():.3f}, std: {scores.std():.3f}")

    return assay_mapping


def validate_virogym(data_dir: Path) -> dict:
    """
    Validate ViroGym data (GSK-AI/viroGym GitHub repository).

    Expected structure:
        data/ViroGym/
        ├── DMS/
        │   ├── benchmark.csv
        │   └── cleaned_benchmark/*.csv
        ├── neutralization/
        │   └── cleaned_benchmark/*.csv
        └── GISAID/
            └── cleaned_benchmark/*.csv

    Returns a dict with file paths and statistics for config.yaml.
    """
    logger.info("\n" + "=" * 50)
    logger.info("Validating ViroGym data...")
    logger.info("=" * 50)

    virogym_dir = data_dir / "ViroGym"
    status = {}

    # --- Key files to check ---
    checks = {
        "SARS2_Spike_RBD": {
            "escape": "DMS/cleaned_benchmark/SARS_antibody_escape_Wuhan_Hu_1.csv",
            "binding": "DMS/cleaned_benchmark/SARS_binding_Wuhan_Hu_1_RBD.csv",
        },
        "H5N1_HA": {
            "cell_entry": "DMS/cleaned_benchmark/FLU_cell_entry_H5N1.csv",
        },
    }

    for protein_id, paths in checks.items():
        logger.info(f"\n{protein_id}:")
        for label, rel_path in paths.items():
            full = virogym_dir / rel_path
            if full.exists():
                df = pd.read_csv(full)
                scores = df["DMS_score"].dropna()
                logger.info(f"  [{label}] {rel_path}")
                logger.info(f"    Rows: {len(df):,}")
                logger.info(f"    Columns: {list(df.columns)}")
                logger.info(f"    Scores: [{scores.min():.3f}, {scores.max():.3f}], "
                            f"mean={scores.mean():.3f}, std={scores.std():.3f}")
                status[f"{protein_id}_{label}"] = {
                    "file": rel_path, "rows": len(df),
                    "columns": list(df.columns),
                    "score_range": [float(scores.min()), float(scores.max())],
                }
            else:
                logger.warning(f"  [{label}] MISSING: {rel_path}")

    # --- DMS benchmark index ---
    bench_path = virogym_dir / "DMS" / "benchmark.csv"
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        logger.info(f"\nDMS benchmark index: {len(bench)} assays")
        logger.info(f"  Assay types: {bench['phenotype_category'].value_counts().to_dict()}")
        logger.info(f"  Viruses:     {bench['virus'].value_counts().to_dict()}")

    # --- Neutralisation ---
    neut_dir = virogym_dir / "neutralization" / "cleaned_benchmark"
    if neut_dir.exists():
        neut_files = list(neut_dir.glob("*.csv"))
        logger.info(f"\nNeutralisation assays: {len(neut_files)} files")
        for f in sorted(neut_files)[:3]:
            df = pd.read_csv(f)
            logger.info(f"  {f.name}: {len(df)} rows, cols={list(df.columns)}")
        if len(neut_files) > 3:
            logger.info(f"  ... and {len(neut_files) - 3} more")

    return status


def generate_config_patch(assay_mapping: dict, virogym_mapping: dict) -> str:
    """
    Generate a YAML snippet to patch into config.yaml with actual assay IDs
    and column names discovered during validation.
    """
    lines = []
    lines.append("# ============================================================")
    lines.append("# Auto-generated config patch from data_prep.py")
    lines.append("# Copy these values into config.yaml under the appropriate")
    lines.append("# scenario sections.")
    lines.append("# ============================================================")
    lines.append("")

    if assay_mapping:
        lines.append("# ProteinGym assay IDs:")
        for protein_id, assay_id in assay_mapping.items():
            lines.append(f"#   {protein_id}: {assay_id}")
        lines.append("")

    if virogym_mapping:
        lines.append("# ViroGym file mappings:")
        for protein_id, info in virogym_mapping.items():
            lines.append(f"#   {protein_id}: file={info['file']}, columns={info['columns']}")
        lines.append("")

    return "\n".join(lines)


def create_data_index(data_dir: Path) -> None:
    """Create a simple index file summarizing available data."""
    index = {
        "created": pd.Timestamp.now().isoformat(),
        "datasets": {},
    }

    proteingym_csv = data_dir / "ProteinGym" / "DMS_substitutions.csv"
    if proteingym_csv.exists():
        df = pd.read_csv(proteingym_csv)
        index["datasets"]["ProteinGym"] = {
            "file": "DMS_substitutions.csv",
            "rows": len(df),
            "columns": list(df.columns),
            "n_assays": int(df["DMS_id"].nunique()) if "DMS_id" in df.columns else 0,
        }

    for f in (data_dir / "ViroGym").glob("*.csv"):
        df = pd.read_csv(f)
        index["datasets"][f"ViroGym/{f.name}"] = {
            "file": f"ViroGym/{f.name}",
            "rows": len(df),
            "columns": list(df.columns),
        }

    index_path = data_dir / "data_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"\nData index saved to {index_path}")


def main():
    parser = argparse.ArgumentParser(description="DRYLAB-Bench Data Preparation")
    parser.add_argument("--data-dir", default="./data", help="Path to data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Validate ProteinGym
    pg_mapping = validate_proteingym(data_dir)

    # Validate ViroGym
    vg_mapping = validate_virogym(data_dir)

    # Generate config patch
    patch = generate_config_patch(pg_mapping, vg_mapping)
    patch_path = data_dir / "config_patch.yaml"
    with open(patch_path, "w") as f:
        f.write(patch)
    logger.info(f"\nConfig patch written to {patch_path}")

    # Create data index
    create_data_index(data_dir)

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Data preparation complete.")
    logger.info("=" * 50)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review the config patch at data/config_patch.yaml")
    logger.info("  2. Update config.yaml with the discovered assay IDs and column names")
    logger.info("  3. Run: python run_experiment.py --config config.yaml")
    logger.info("")
    if not pg_mapping:
        logger.warning("⚠  No ProteinGym assays were auto-detected.")
        logger.warning("   You may need to manually specify assay IDs in config.yaml.")
    if not vg_mapping:
        logger.warning("⚠  No ViroGym data files were found.")
        logger.warning("   Download ViroGym data or use ProteinGym-only scenarios.")


if __name__ == "__main__":
    main()
