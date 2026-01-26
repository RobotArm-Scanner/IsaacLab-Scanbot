# Re-enable IsaacLab aliases only when deactivating the last conda env
if [ "${CONDA_SHLVL:-0}" = "1" ]; then
  if [ -n "${ISAACLAB_PATH:-}" ] && [ -x "${ISAACLAB_PATH}/_isaac_sim/python.sh" ]; then
    alias python="${ISAACLAB_PATH}/_isaac_sim/python.sh"
    alias python3="${ISAACLAB_PATH}/_isaac_sim/python.sh"
    alias pip="${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip"
    alias pip3="${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip"
    if [ -x "${ISAACLAB_PATH}/_isaac_sim/tensorboard" ]; then
      alias tensorboard="${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/_isaac_sim/tensorboard"
    fi
  fi
fi
