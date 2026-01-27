# Scanbot RL Spec

## Goal
Train Scanbot to scan teeth with cumulative coverage while minimizing motion.

## Environment
- Task id: `e2.t3ds.rl`
- Base: ManagerBasedRLEnv (Piper no-gripper, Scanbot E2)
- Episode length: configurable (`episode_length_s`), default kept under 3 minutes
- Envs: 1-4 (override via `--num_envs` in training or `SCANBOT_NUM_ENVS` for launcher)

## Observations (policy)
- joint_pos, joint_vel
- ee_pos, ee_quat
- action history disabled
- camera obs intentionally disabled for stability; can be enabled later

## Actions
- Joint position commands for 6 DOF arm

## Reward (per step)
Let:
- `C_t = teeth.all.coverage + teeth_gum.all.coverage`
- `d_ee = ||ee_pos_t - ee_pos_{t-1}||`

Reward terms (weights configurable in cfg):
1) Movement penalty:
   - `r_move = -w_move * d_ee`
2) Coverage delta:
   - `r_cov = +w_cov * (C_t - C_{t-1})`
3) Per-tooth bonus (one-time per tooth):
   - `r_tooth = +w_tooth * count_new_teeth(coverage >= threshold_tooth)`
4) Total bonus (one-time per episode):
   - `r_total = +w_total * 1[C_t >= threshold_total]`

Thresholds are config parameters (not hard-coded).

## Coverage Pipeline
1) Depth from wrist camera (`distance_to_image_plane`)
2) Depth -> pointcloud via `isaaclab.sensors.camera.utils.create_pointcloud_from_depth`
3) Transform points to teeth local frame
4) Voxel downsample and point cap
5) Update `CoverageTracker` (KDTree over cached surface samples)
6) Update coverage at a configurable cadence (`coverage_update_every`)

## Caching (teeth3ds_utils.py)
- Precompute surface samples from OBJ+JSON labels
- Cache stored at `resources/teeth/t3ds/cache/*.npz`
- Cache key includes dataset_id, num_samples, seed, scale, gum_assign_radius
- KDTree built once per run

## Training
Suggested entry point:
```
export DISPLAY=:3
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task e2.t3ds.rl --num_envs 2 --max_iterations 50 \
  --enable_cameras --kit_args "--enable omni.usd.metrics.assembler"
```

Increase `num_envs` and iterations after stability is confirmed.
