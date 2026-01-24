MULTI_GPU_ENABLED="${SCANBOT_MULTIGPU_ENABLED:-true}"
MULTI_GPU_AUTO="${SCANBOT_MULTIGPU_AUTO:-true}"
MULTI_GPU_MAX="${SCANBOT_MULTIGPU_MAX:-8}"
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
KIT_ARGS="--/renderer/multiGpu/enabled=${MULTI_GPU_ENABLED} --/renderer/multiGpu/autoEnable=${MULTI_GPU_AUTO} --/renderer/multiGpu/maxGpuCount=${MULTI_GPU_MAX} ${ACTIVE_GPU_ARG} ${PHYSX_GPU_ARG} --/app/renderer/waitIdle=false --/app/hydraEngine/waitIdle=false --/app/updateOrder/checkForHydraRenderComplete=0 --enable omni.usd.metrics.assembler"
NUM_ENVS="${SCANBOT_NUM_ENVS:-1}"

script -q -f -e -a \
  /workspace/isaaclab/scanbot/logs/isaaclab.log \
  -c "bash -lc './isaaclab.sh -p scanbot/scripts/launchers/basic_launcher.py \
        --ext-folder scanbot/extensions \
        --enable_cameras \
        --num_envs ${NUM_ENVS} \
        ${DEVICE_ARG} \
        --kit_args \"${KIT_ARGS}\"'"
