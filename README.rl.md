# Scanbot RL 실행 방법

## 용어
- 호스트: 이 GPU 서버
- PROJECT_SUFFIX: `docker/.env.scanbot`의 프로젝트 식별자
- DOCKER_NAME_SUFFIX: `docker/.env.scanbot`의 DOCKER_NAME_SUFFIX
- 컨테이너: `isaac-lab-scanbot${DOCKER_NAME_SUFFIX}`

## 0. 사전 준비
1) `docker/.env.scanbot`에서 `PROJECT_SUFFIX` 설정
2) `DISPLAY` 설정
   - `docker/.env.scanbot`에 `DISPLAY`가 없으면 호스트의 사용 중인 X display(`:0`, `:59` 등)를 확인해 추가
3) 컨테이너 시작
```bash
docker/container.sh start scanbot --no-build
```
4) 컨테이너 진입(예시)
```bash
docker/container.sh enter scanbot
# 또는
# docker exec -it isaac-lab-scanbot${DOCKER_NAME_SUFFIX} bash
```

## 1. tmux 세션
```bash
tmux ls
# 없으면
# tmux new -s isaaclab${DOCKER_NAME_SUFFIX}
# 들어가려면
# tmux attach -t isaaclab${DOCKER_NAME_SUFFIX}
```
- IsaacLab 실행은 반드시 `isaaclab${DOCKER_NAME_SUFFIX}` 세션에서만 수행

## 2. 기존 프로세스 확인/종료
컨테이너 내부에서 실행 중인 학습 프로세스가 있는지 먼저 확인합니다.
```bash
ps -eo pid,cmd | grep -E "train.py|scanbot_launcher.py" | grep -v grep
```
종료가 필요하면:
```bash
# tmux 세션에서
Ctrl+C
# 종료가 안 되면
Ctrl+Z
kill %1
# 또는
pkill -f train.py
```

## 3. RL 학습 실행 (GUI 모드)
```bash
cd /workspace/isaaclab
set -a && source docker/.env.scanbot && set +a
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task e2.t3ds.rl \
  --num_envs 4 \
  --max_iterations 200 \
  --enable_cameras \
  --kit_args "--enable omni.usd.metrics.assembler"
```

## 4. RL 학습 실행 (헤드리스)
```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task e2.t3ds.rl \
  --num_envs 4 \
  --max_iterations 200 \
  --headless \
  --enable_cameras \
  --kit_args "--enable omni.usd.metrics.assembler"
```

## 5. 로그/체크포인트
- 학습 로그: `logs/rsl_rl/scanbot_e2_t3ds_rl/<timestamp>`
- 기본 설정: `scanbot/scripts/rl/rsl_rl_ppo_cfg.py`
- 재개 학습 예시:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task e2.t3ds.rl \
  --resume \
  --load_run <run_dir> \
  --checkpoint <checkpoint_file>
```

## 6. 자주 쓰는 옵션
- `--num_envs`: 병렬 환경 수
- `--max_iterations`: 학습 iteration 수
- `--seed`: 시드 고정
- `--device cuda:0`: 장치 지정
