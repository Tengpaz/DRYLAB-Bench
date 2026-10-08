#!/usr/bin/env python3
"""
DRYLAB-Bench — Dataset Downloader

Downloads ProteinGym v1 from HuggingFace (5 Parquet shards, ~800 MB total)
and merges them into a single CSV file.

Usage:
    python download_datasets.py --data-dir ./data
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


# ==========================================================================
# Download with retry + resume + progress bar
# ==========================================================================

def _download_file_with_resume(url: str, local_path: Path, max_retries: int = 5) -> None:
    """Download a file with resume support and retry logic."""
    import requests

    local_path.parent.mkdir(parents=True, exist_ok=True)
    existing = local_path.stat().st_size if local_path.exists() else 0

    for attempt in range(1, max_retries + 1):
        headers = {}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=(30, 300))
        except (requests.ConnectionError, requests.ReadTimeout) as e:
            print(f"    {e} (attempt {attempt}/{max_retries}), retrying in {min(2**attempt,30)}s...")
            time.sleep(min(2 ** attempt, 30))
            continue

        if resp.status_code not in (200, 206):
            print(f"    HTTP {resp.status_code} (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            resp.raise_for_status()

        total = existing + int(resp.headers.get("content-length", 0) or 0)
        mode = "ab" if resp.status_code == 206 else "wb"
        if resp.status_code == 200:
            existing = 0

        try:
            from tqdm import tqdm
            pbar = tqdm(
                total=total, initial=existing, unit="B",
                unit_scale=True, unit_divisor=1024,
                desc=f"    {local_path.name}",
            )
        except ImportError:
            pbar = None

        try:
            with open(local_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        existing += len(chunk)
                        if pbar:
                            pbar.update(len(chunk))
            if pbar:
                pbar.close()
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"    OK ({size_mb:.0f} MB)")
            return
        except Exception as e:
            if pbar:
                pbar.close()
            print(f"    Interrupted: {e} — {existing/(1024*1024):.0f} MB saved, will resume.")
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Failed after {max_retries} attempts.")


# ==========================================================================
# Main download logic
# ==========================================================================

REPO = "OATML-Markslab/ProteinGym_v1"
RESOLVE_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

SHARDS = [
    "DMS_substitutions/train-00000-of-00005.parquet",
    "DMS_substitutions/train-00001-of-00005.parquet",
    "DMS_substitutions/train-00002-of-00005.parquet",
    "DMS_substitutions/train-00003-of-00005.parquet",
    "DMS_substitutions/train-00004-of-00005.parquet",
]


def download_proteingym(target_dir: Path) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = target_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)

    print("Downloading DMS_substitutions shards (5 files, ~800 MB total)...")
    print("  Source:  https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1")
    print("  Resume:  safe to Ctrl+C and re-run — progress is preserved.")
    print()

    paths = []

    for i, filename in enumerate(SHARDS):
        local_path = raw_dir / filename.replace("/", "_")
        url = f"{RESOLVE_BASE}/{filename}"
        print(f"  [{i + 1}/5] {filename}")

        # Skip if already complete (assume >10 MB = complete enough)
        if local_path.exists() and local_path.stat().st_size > 10 * 1024 * 1024:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"    Already downloaded ({size_mb:.0f} MB), skipping.")
            paths.append(local_path)
            continue

        _download_file_with_resume(url, local_path)
        paths.append(local_path)

    # Merge shards → CSV
    import pandas as pd

    csv_path = target_dir / "DMS_substitutions.csv"
    if csv_path.exists():
        print(f"\n  {csv_path.name} already exists, skipping merge.")
    else:
        print(f"\nMerging {len(paths)} shards into DMS_substitutions.csv ...")
        dfs = []
        for p in paths:
            df = pd.read_parquet(p)
            dfs.append(df)
            print(f"  {p.name}: {len(df):,} rows")
        merged = pd.concat(dfs, ignore_index=True)
        print(f"  Total: {len(merged):,} rows, {len(merged.columns)} columns")
        print(f"  Writing {csv_path} ...")
        merged.to_csv(csv_path, index=False)
        print(f"  Done ({csv_path.stat().st_size / (1024*1024):.0f} MB)")

    # Cleanup
    print(f"  Cleaning up raw Parquet shards...")
    shutil.rmtree(raw_dir, ignore_errors=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    target_dir = Path(args.data_dir) / "ProteinGym"

    print("=" * 60)
    print("  DRYLAB-Bench — ProteinGym v1 Download")
    print("=" * 60)
    print()

    ok = download_proteingym(target_dir)

    print()
    csv_path = target_dir / "DMS_substitutions.csv"
    if csv_path.exists():
        print(f"  [OK] DMS_substitutions.csv — {csv_path.stat().st_size / (1024*1024):.0f} MB")
        print(f"  Next: python data_prep.py --data-dir ./data")
    else:
        print("  [FAIL] Download incomplete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
