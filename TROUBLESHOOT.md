# 트러블슈팅 노트

## 초기화 시 카메라 depth 출력 크기가 0
**증상**
- env 초기화 중 아래 에러로 크래시:
  `RuntimeError: shape '[H, W, 1]' is invalid for input of size 0`
  (`isaaclab/sensors/camera/camera.py::_process_annotator_output`)

**원인**
- Replicator가 준비되기 전에(ObservationManager 초기화 단계) 카메라 데이터를 접근함.

**해결**
- 초기화 단계에서 카메라 관측을 호출하지 않기(Policy group에서 image obs 제거).
- 카메라 cfg에는 depth 설정을 유지하되, sim 시작 이후에 접근.
- 특정 프레임에서 depth가 비어있으면 해당 step의 커버리지 업데이트를 스킵.

## cfg의 num_envs와 실제 num_envs가 다름
**증상**
- cfg에서 더 큰 값을 설정했는데도 `num_envs = 1`로 동작함.

**원인**
- `scanbot.sh`가 `--num_envs ${SCANBOT_NUM_ENVS:-1}`를 전달하면서 cfg 값을 override함.

**해결**
- `scanbot.sh` 실행 전에 `SCANBOT_NUM_ENVS=2`(원하는 값)로 설정하거나,
- 학습 스크립트에 `--num_envs`를 직접 전달.

## Isaac Sim 시작이 멈추거나 응답 없음
**증상**
- Isaac Sim이 완전히 올라오지 않거나 멈춘 것처럼 보임.

**해결**
- tmux에서 프로세스를 중지(Ctrl+C)하거나, 컨테이너에서 강제 종료:
  `pkill -f scanbot_launcher.py`
- tmux 세션(`isaaclab${DOCKER_NAME_SUFFIX}`; 예: `isaaclab-dev`)에서 다시 실행.

## RL 학습: env가 등록되지 않음
**증상**
- `gymnasium.error.NameNotFound: Environment e2.t3ds.rl doesn't exist.`

**해결**
- `isaaclab_tasks`가 자동 등록할 수 있도록 `scanbot_task`를 import하도록 `isaaclab_tasks.manager_based.scanbot` 모듈을 추가.

## RL 학습: omni.usd.metrics.assembler 누락
**증상**
- `ModuleNotFoundError: No module named 'omni.metrics'`

**해결**
- kit args를 전달:
  `--kit_args "--enable omni.usd.metrics.assembler"`

## RL 학습: 카메라 사용 시 enable_cameras 필요
**증상**
- `RuntimeError: A camera was spawned without the --enable_cameras flag.`

**해결**
- 학습/런치 시 `--enable_cameras`를 전달.

## RL 학습: headless에서 GLXBadFBConfig
**증상**
- 카메라 사용 + `--headless` 조합에서 `GLXBadFBConfig` 발생.

**해결**
- `DISPLAY=:3` 등으로 non-headless로 실행하거나, EGL을 올바르게 구성.

## RL 학습: actor_critic이 1D obs를 기대
**증상**
- `AssertionError: The ActorCritic module only supports 1D observations.`

**해결**
- `ScanbotRLObservationsCfg` 사용 + `policy.concatenate_terms = True`로 관측을 flatten.

## RL 학습: hydra가 OpenXR lambda를 직렬화 못함
**증상**
- `ValueError: Could not resolve the input string 'lambda headpose' into callable object.`

**해결**
- RL cfg에서 `xr.anchor_rotation_custom_func = None`으로 설정하고 `teleop_devices` 비활성화.

## RL 학습: env 수가 많을 때 PhysX GPU 커널 실패
**증상**
- 로그에 `PhysX error: ... fail to launch kernel`, `PhysX has reported too many errors, simulation has been stopped.` 출력.

**해결**
- `--num_envs`를 줄이기(벤치에서 env8은 실패했고 env1~4는 안정적).
- 렌더/카메라 부하를 줄이거나 추가 센서를 비활성화하는 것도 고려.
