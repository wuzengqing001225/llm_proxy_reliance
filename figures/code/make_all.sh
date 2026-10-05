#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 figures/code/make_paper_figures.py "$@"
