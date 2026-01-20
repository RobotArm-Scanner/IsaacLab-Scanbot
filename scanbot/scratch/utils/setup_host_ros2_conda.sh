#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-scanbot-ros2}"
CHANNELS=("-c" "conda-forge" "-c" "robostack-staging")
PACKAGES=(
  "python=3.10"
  "ros-humble-ros-base"
  "ros-humble-ros2cli"
  "colcon-common-extensions"
)

if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN=mamba
elif command -v micromamba >/dev/null 2>&1; then
  CONDA_BIN=micromamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN=conda
else
  echo "[ERROR] conda/mamba not found. Install Miniconda/Miniforge or mamba first." >&2
  exit 1
fi

"${CONDA_BIN}" create -y -n "${ENV_NAME}" "${CHANNELS[@]}" "${PACKAGES[@]}"

echo "[INFO] Created env: ${ENV_NAME}"
echo "[INFO] Next: conda activate ${ENV_NAME}"
