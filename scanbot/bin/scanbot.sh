export DISPLAY=:3
script -q -f -e -a \
  /workspace/isaaclab/scanbot/logs/isaaclab.log \
  -c "bash -lc 'if [ -f /opt/ros/humble/setup.bash ]; then source /opt/ros/humble/setup.bash; fi; \
      if [ -f /workspace/isaaclab/scanbot/ros2/install/setup.bash ]; then source /workspace/isaaclab/scanbot/ros2/install/setup.bash; fi; \
      ./isaaclab.sh -p scanbot/scripts/launchers/basic_launcher.py \
        --ext-folder scanbot/extensions \
        --enable_cameras \
        --num_envs 1 \
        --kit_args \"--/renderer/multiGpu/enabled=true --/renderer/multiGpu/autoEnable=true --/renderer/multiGpu/maxGpuCount=8 --enable omni.usd.metrics.assembler\"'"
