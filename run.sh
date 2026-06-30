#!/usr/bin/env bash
# NYC Price Predictor — launcher script
# Usage:
#   ./run.sh notebook   → open Jupyter notebook in browser
#   ./run.sh predict    → run standalone predict.py (faster, no Jupyter needed)
#   ./run.sh install    → (re)install all dependencies
#   ./run.sh            → defaults to 'notebook'

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-notebook}"

install_deps() {
    echo "Installing dependencies..."
    pip install jupyter notebook pandas numpy matplotlib seaborn \
                scikit-learn xgboost lightgbm plotly --break-system-packages
    echo "Done."
}

case "$MODE" in
    install)
        install_deps
        ;;
    predict)
        echo "Running standalone predictor..."
        python predict.py
        ;;
    notebook)
        echo "Starting Jupyter Notebook..."
        echo "Open  http://localhost:8888  in your browser"
        echo "Press Ctrl+C to stop."
        jupyter notebook nyc_price_analysis.ipynb
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./run.sh [notebook|predict|install]"
        exit 1
        ;;
esac
