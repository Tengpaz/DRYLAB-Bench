#!/bin/bash
# ============================================================================
# DRYLAB-Bench — Dataset Download Script
# ============================================================================
# Thin wrapper: delegates to download_datasets.py (Python, more reliable).
#
# Usage:
#   bash download_datasets.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m drylab_bench.download_datasets --data-dir ./data "$@"
