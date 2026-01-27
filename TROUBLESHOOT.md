# Troubleshooting Notes

## Camera depth output size 0 at init
**Symptoms**
- Crash during env init with:
  `RuntimeError: shape '[H, W, 1]' is invalid for input of size 0`
  from `isaaclab/sensors/camera/camera.py::_process_annotator_output`.

**Cause**
- Camera data is accessed during ObservationManager init (before replicator is ready).

**Fix**
- Do not call camera observations during init (remove image obs from policy group).
- Keep depth in camera config but access it after sim starts.
- If depth is still empty for a frame, skip coverage update for that step.

## num_envs not matching cfg
**Symptoms**
- Env shows `num_envs = 1` even if cfg sets a higher value.

**Cause**
- `scanbot.sh` passes `--num_envs ${SCANBOT_NUM_ENVS:-1}` which overrides cfg.

**Fix**
- Set `SCANBOT_NUM_ENVS=2` (or desired value) before running `scanbot.sh`, or
- Pass `--num_envs` to the training script.

## Isaac Sim startup hang
**Symptoms**
- Isaac Sim does not fully start or becomes unresponsive.

**Fix**
- Stop the process from tmux (Ctrl+C) or kill in container:
  `pkill -f scanbot_launcher.py`
- Restart via tmux session `isaaclab`.

## RL training: env not registered
**Symptoms**
- `gymnasium.error.NameNotFound: Environment e2.t3ds.rl doesn't exist.`

**Fix**
- Added `isaaclab_tasks.manager_based.scanbot` module to import `scanbot_task` so `isaaclab_tasks` auto-registers.

## RL training: missing omni.usd.metrics.assembler
**Symptoms**
- `ModuleNotFoundError: No module named 'omni.metrics'`

**Fix**
- Pass kit args: `--kit_args "--enable omni.usd.metrics.assembler"`

## RL training: camera requires enable_cameras
**Symptoms**
- `RuntimeError: A camera was spawned without the --enable_cameras flag.`

**Fix**
- Pass `--enable_cameras` to training/launch.

## RL training: GLXBadFBConfig in headless
**Symptoms**
- `GLXBadFBConfig` when using `--headless` with cameras.

**Fix**
- Run non-headless with `DISPLAY=:3`, or configure EGL properly.

## RL training: actor_critic expects 1D obs
**Symptoms**
- `AssertionError: The ActorCritic module only supports 1D observations.`

**Fix**
- Use `ScanbotRLObservationsCfg` with `policy.concatenate_terms = True` to flatten obs.

## RL training: hydra cannot serialize OpenXR lambda
**Symptoms**
- `ValueError: Could not resolve the input string 'lambda headpose' into callable object.`

**Fix**
- In RL config, set `xr.anchor_rotation_custom_func = None` and disable `teleop_devices`.

## RL training: PhysX GPU kernel failures at high env counts
**Symptoms**
- Logs show `PhysX error: ... fail to launch kernel` and `PhysX has reported too many errors, simulation has been stopped.`

**Fix**
- Reduce `--num_envs` (env8 failed in our bench; env1-4 were stable).
- Consider lowering render/camera load or disabling extra sensors.
