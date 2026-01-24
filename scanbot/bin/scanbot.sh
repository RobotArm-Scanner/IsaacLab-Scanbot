HEADLESS_FLAG="${SCANBOT_HEADLESS:-0}"
HEADLESS_GLX_FALLBACK="${SCANBOT_HEADLESS_GLX_FALLBACK:-1}"
LIVESTREAM_FLAG="${SCANBOT_LIVESTREAM:-}"
if [ -n "${LIVESTREAM_FLAG}" ]; then
  export LIVESTREAM="${LIVESTREAM_FLAG}"
fi
LIVESTREAM_PORT_ARG=""
if [ -n "${SCANBOT_LIVESTREAM_PORT:-}" ]; then
  LIVESTREAM_PORT_ARG="--/app/livestream/port=${SCANBOT_LIVESTREAM_PORT}"
fi
EGL_AVAILABLE=0
for egl_path in /isaac-sim/kit/kernel/plugins/libcarb.windowing-egl.plugin.so /isaac-sim/extscache/carb.windowing.plugins-*/bin/libcarb.windowing-egl.plugin.so; do
  if [ -f "${egl_path}" ]; then
    EGL_AVAILABLE=1
    break
  fi
done

case "${HEADLESS_FLAG}" in
  1|true|TRUE|yes|YES)
    if [ "${EGL_AVAILABLE}" != "1" ] && [ "${HEADLESS_GLX_FALLBACK}" != "0" ]; then
      echo "[WARN] EGL windowing plugin not found; falling back to GLX windowed mode for rendering."
      HEADLESS_FLAG=0
      FORCE_WINDOW="${SCANBOT_FORCE_WINDOW:-1}"
      if [ -z "${SCANBOT_DISPLAY}" ]; then
        SCANBOT_DISPLAY=":1"
      fi
    fi
    ;;
esac

case "${HEADLESS_FLAG}" in
  1|true|TRUE|yes|YES)
    export HEADLESS=1
    if [ "${SCANBOT_HEADLESS_KEEP_DISPLAY:-1}" = "1" ]; then
      export DISPLAY="${SCANBOT_DISPLAY:-:3}"
    else
      unset DISPLAY
    fi
    ;;
  *)
    export DISPLAY="${SCANBOT_DISPLAY:-:3}"
    ;;
esac
MULTI_GPU_ENABLED="${SCANBOT_MULTIGPU_ENABLED:-true}"
MULTI_GPU_AUTO="${SCANBOT_MULTIGPU_AUTO:-true}"
MULTI_GPU_MAX="${SCANBOT_MULTIGPU_MAX:-8}"
FORCE_WINDOW="${FORCE_WINDOW:-${SCANBOT_FORCE_WINDOW:-0}}"
WINDOW_ARGS=""
case "${FORCE_WINDOW}" in
  1|true|TRUE|yes|YES)
    WINDOW_ARGS="--/app/window/enabled=true --/app/window/noDecorations=true --/app/window/width=32 --/app/window/height=32 --/app/window/x=-2000 --/app/window/y=-2000"
    ;;
esac
ACTIVE_GPU="${SCANBOT_ACTIVE_GPU:-}"
PHYSX_GPU="${SCANBOT_PHYSX_GPU:-}"
DEVICE_ARG=""
if [ -n "${SCANBOT_DEVICE:-}" ]; then
  DEVICE_ARG="--device ${SCANBOT_DEVICE}"
fi
ACTIVE_GPU_ARG=""
PHYSX_GPU_ARG=""
if [ -n "${ACTIVE_GPU}" ]; then
  ACTIVE_GPU_ARG="--/renderer/activeGpu=${ACTIVE_GPU}"
fi
if [ -n "${PHYSX_GPU}" ]; then
  PHYSX_GPU_ARG="--/physics/cudaDevice=${PHYSX_GPU}"
fi
KIT_ARGS="--/renderer/multiGpu/enabled=${MULTI_GPU_ENABLED} --/renderer/multiGpu/autoEnable=${MULTI_GPU_AUTO} --/renderer/multiGpu/maxGpuCount=${MULTI_GPU_MAX} ${ACTIVE_GPU_ARG} ${PHYSX_GPU_ARG} --/app/renderer/waitIdle=false --/app/hydraEngine/waitIdle=false --/app/updateOrder/checkForHydraRenderComplete=0 ${LIVESTREAM_PORT_ARG} ${WINDOW_ARGS} --enable omni.usd.metrics.assembler"
ENABLE_CAMERAS_FLAG="${SCANBOT_ENABLE_CAMERAS:-1}"
ENABLE_CAMERAS_ARG="--enable_cameras"
case "${ENABLE_CAMERAS_FLAG}" in
  0|false|FALSE|no|NO)
    ENABLE_CAMERAS_ARG=""
    ;;
esac
NUM_ENVS="${SCANBOT_NUM_ENVS:-1}"

script -q -f -e -a \
  /workspace/isaaclab/scanbot/logs/isaaclab.log \
  -c "bash -lc 'if [ -f /opt/ros/humble/setup.bash ]; then source /opt/ros/humble/setup.bash; fi; \
      if [ -f /workspace/isaaclab/scanbot/ros2/install/setup.bash ]; then source /workspace/isaaclab/scanbot/ros2/install/setup.bash; fi; \
      ./isaaclab.sh -p scanbot/scripts/launchers/basic_launcher.py \
        --ext-folder scanbot/extensions \
        ${ENABLE_CAMERAS_ARG} \
        --num_envs ${NUM_ENVS} \
        ${DEVICE_ARG} \
        --kit_args \"${KIT_ARGS}\"'"
