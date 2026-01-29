# Scanbot RL 명세

## 목표
Scanbot이 치아를 스캔하면서 누적 커버리지를 높이고, 동시에 불필요한 움직임을 최소화하도록 학습합니다.

## 환경
- task id: `e2.t3ds.rl`
- 기반: `ManagerBasedRLEnv` (Piper no-gripper, Scanbot E2)
- 에피소드 길이: 설정 가능 (`episode_length_s`), 기본은 3분 이하 유지
- env 개수: 1~4 권장 (학습 시 `--num_envs` 또는 런처에서 `SCANBOT_NUM_ENVS`로 override)

## 관측 (policy)
- `joint_pos`, `joint_vel`
- `ee_pos`, `ee_quat`
- action history 비활성화
- 안정성 우선으로 camera obs는 의도적으로 비활성화(필요 시 추후 활성화)

## 액션
- 6 DOF 팔의 joint position command

## 보상 (스텝당)
정의:
- `C_t = teeth.all.coverage + teeth_gum.all.coverage`
- `d_ee = ||ee_pos_t - ee_pos_{t-1}||`

보상 항(가중치는 cfg에서 조절):
1) 움직임 패널티:
   - `r_move = -w_move * d_ee`
2) 커버리지 증가량:
   - `r_cov = +w_cov * (C_t - C_{t-1})`
3) 치아별 보너스(치아당 1회):
   - `r_tooth = +w_tooth * count_new_teeth(coverage >= threshold_tooth)`
4) 전체 보너스(에피소드당 1회):
   - `r_total = +w_total * 1[C_t >= threshold_total]`

threshold는 cfg 파라미터로 관리하며 하드코딩하지 않습니다.

## 커버리지 파이프라인
1) wrist camera에서 depth(`distance_to_image_plane`) 획득
2) depth → pointcloud 변환: `isaaclab.sensors.camera.utils.create_pointcloud_from_depth`
3) point들을 teeth local frame으로 변환
4) voxel downsample + max point cap 적용
5) `CoverageTracker` 업데이트(KDTree: cached surface samples 기반)
6) 설정된 주기(`coverage_update_every`)로 커버리지 업데이트

## 캐싱 (teeth3ds_utils.py)
- OBJ + JSON label 기반으로 surface sample을 사전 계산
- 캐시 위치: `resources/teeth/t3ds/cache/*.npz`
- 캐시 키: `dataset_id`, `num_samples`, `seed`, `scale`, `gum_assign_radius` 포함
- KDTree는 실행당 1회 생성

## 학습
권장 실행 예시:
```bash
export DISPLAY=:3
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task e2.t3ds.rl --num_envs 2 --max_iterations 50 \
  --enable_cameras --kit_args "--enable omni.usd.metrics.assembler"
```

실행 세부 절차는 `README.rl.md`를 참고하세요.

안정성이 확인된 뒤에 `num_envs`와 iteration을 점진적으로 늘립니다.
