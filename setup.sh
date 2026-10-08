#!/bin/bash
# ============================================================================
# DRYLAB-Bench — One-click Environment Setup
# ============================================================================
# Creates conda environment, installs dependencies, downloads datasets,
# and validates data readiness.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Options:
#   ./setup.sh --env-only        Only create conda env (skip data download)
#   ./setup.sh --data-only       Only download datasets (skip conda env)
#   ./setup.sh --help            Show this help
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="drylab_bench"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step()  { echo -e "\n${GREEN}[==]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DO_ENV=true
DO_DATA=true

for arg in "$@"; do
    case $arg in
        --env-only)  DO_DATA=false ;;
        --data-only) DO_ENV=false ;;
        --help)      head -30 "$0"; exit 0 ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Conda environment
# ---------------------------------------------------------------------------
if $DO_ENV; then
    step "Setting up conda environment: $ENV_NAME"

    # Check conda availability
    if ! command -v conda &>/dev/null; then
        error "conda not found. Please install Miniconda or Anaconda first."
        echo "  macOS:  brew install miniconda"
        echo "  Linux:  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    CONDA_VERSION=$(conda --version 2>/dev/null || echo "unknown")
    echo "  conda version: $CONDA_VERSION"

    # Remove existing environment if it exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        warn "Environment '$ENV_NAME' already exists."
        read -rp "  Remove and recreate? [y/N] " answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            conda env remove -n "$ENV_NAME" -y
        else
            echo "  Skipping environment creation."
            DO_ENV=false
        fi
    fi

    if $DO_ENV; then
        echo "  Creating environment from environment.yml..."
        conda env create -f "$SCRIPT_DIR/environment.yml" -y

        echo ""
        echo "  Environment created successfully."
        echo "  Activate with:  conda activate $ENV_NAME"
    fi
else
    echo "Skipping conda environment setup (--data-only)."
fi

# ---------------------------------------------------------------------------
# Step 2: Install the package in development mode
# ---------------------------------------------------------------------------
if $DO_ENV; then
    step "Installing DRYLAB-Bench in development mode"

    # Need to source conda first
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "$ENV_NAME" 2>/dev/null || {
        warn "Could not activate environment. Trying conda run..."
        CONDA_RUN="conda run -n $ENV_NAME"
    }

    if [[ -n "${CONDA_RUN:-}" ]]; then
        $CONDA_RUN pip install -e "$SCRIPT_DIR" --quiet
    else
        pip install -e "$SCRIPT_DIR" --quiet
    fi
    echo "  Package installed."
fi

# ---------------------------------------------------------------------------
# Step 3: API key configuration
# ---------------------------------------------------------------------------
step "API Key Configuration"

ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    echo "  .env file already exists."
else
    echo "  Creating .env template..."
    cat > "$ENV_FILE" << 'EOF'
# DRYLAB-Bench API Keys
# Uncomment and fill in your keys.

# OpenAI (GPT-4o)
# OPENAI_API_KEY=sk-...

# Anthropic (Claude Opus 4)
# ANTHROPIC_API_KEY=sk-ant-...

# Google (Gemini 2.5 Pro)
# GOOGLE_API_KEY=...

EOF
    echo "  Created $ENV_FILE — please edit it to add your API keys."
    echo ""
    warn "IMPORTANT: You must add your API keys to .env before running experiments."
    echo "  Edit:  $ENV_FILE"
fi

# ---------------------------------------------------------------------------
# Step 4: Dataset download
# ---------------------------------------------------------------------------
if $DO_DATA; then
    step "Downloading datasets"

    if [ -f "$SCRIPT_DIR/download_datasets.sh" ]; then
        bash "$SCRIPT_DIR/download_datasets.sh"
    else
        warn "download_datasets.sh not found — skipping."
    fi
fi

# ---------------------------------------------------------------------------
# Step 5: Data validation
# ---------------------------------------------------------------------------
if $DO_DATA && $DO_ENV; then
    step "Validating datasets"

    if [[ -n "${CONDA_RUN:-}" ]]; then
        $CONDA_RUN python -m drylab_bench.data_prep --data-dir "$SCRIPT_DIR/data"
    else
        python -m drylab_bench.data_prep --data-dir "$SCRIPT_DIR/data"
    fi
fi

# ---------------------------------------------------------------------------
# Step 6: API key source reminder
# ---------------------------------------------------------------------------
step "Setup Complete"

echo ""
echo "  Before running experiments, source your API keys:"
echo "    source $SCRIPT_DIR/.env"
echo ""
echo "  Or export them directly:"
echo "    export OPENAI_API_KEY=sk-..."
echo "    export ANTHROPIC_API_KEY=sk-ant-..."
echo "    export GOOGLE_API_KEY=..."
echo ""
echo "  Activate environment:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Run an experiment:"
echo "    cd $SCRIPT_DIR"
echo "    python run_experiment.py --config config.yaml --data-dir ./data"
echo ""

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  DRYLAB-Bench setup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
