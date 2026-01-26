#!/usr/bin/env bash
set -euo pipefail

ACT_DIR="$PREFIX/etc/conda/activate.d"
DEACT_DIR="$PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACT_DIR" "$DEACT_DIR"

install -m 0644 "$RECIPE_DIR/hooks/00-disable-isaaclab-alias.sh" "$ACT_DIR/00-disable-isaaclab-alias.sh"
install -m 0644 "$RECIPE_DIR/hooks/00-enable-isaaclab-alias.sh" "$DEACT_DIR/00-enable-isaaclab-alias.sh"
