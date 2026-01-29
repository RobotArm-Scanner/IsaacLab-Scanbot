# 시스템
- 본 시스템은 `IsaacSim/IsaacLab(아이작심/아이작랩)` 환경을 실행하기 위해 구성된 GPU 서버임
- 너는 시스템의 `sudo` 권한이 있음 
   - `sudoer`에 등록되지 않았거나, 비밀번호를 물어보는 경우 사용자에게 요청할 것

# 지칭 및 세팅
- 이 시스템을 `호스트(host)`라고 부를 것임
- `docker/.env.scanbot`의 `PROJECT_SUFFIX` 환경변수를 `PROJECT_SUFFIX`라고 부를 것임
- `PROJECT_SUFFIX`는 `프로젝트 식별자`라고 불리기도 함
- `docker/.env.scanbot`의 `DOCKER_NAME_SUFFIX` 환경변수를 `DOCKER_NAME_SUFFIX`라고 부를 것임
- 여기서 언급되는 컨테이너란 `isaac-lab-scanbot${DOCKER_NAME_SUFFIX}` 컨테이너를 말함
- 프로젝트 루트의 `scanbot` 디렉토리는 컨테이너의 `/workspace/isaaclab/scanbot` 디렉토리에 마운트 되어있음 

# IsaacLab 실행 및 종료
- 컨테이너 내부에서 `/workspace/isaaclab/scanbot/bin/scanbot.sh`를 실행하면 IsaacLab이 실행됨
- `scanbot/logs/isaaclab.log`에서 로그를 볼 수 있음
- 나는 보통 tmux 세션 `isaaclab${DOCKER_NAME_SUFFIX}`에서 컨테이너에 진입한 다음에 저 명령을 실행해 (없으면 네가 세션을 만들 것)
- 내가 직접 띄워둔 경우도 있으니 항상 tmux 세션 유무와 현재 쉘이 호스트인지 컨테이너 내부인지 먼저 확인할 것
- IsaacLab 실행 전에는 기존 프로세스가 남아있는지 반드시 확인하고(중복 실행 방지), 종료가 확실히 된 것을 확인한 뒤에만 새로 실행할 것
- IsaacLab은 반드시 tmux 세션 `isaaclab${DOCKER_NAME_SUFFIX}`에서만 띄울 것(백그라운드/분리 실행 금지)
- 이 세션에 `Ctrl + C` 또는 `Ctrl + Z` 이후 `kill %1` (kill을 여러번 보내야 할 수도 있음, 또는 프로세스 검색 후 `kill`)을 쓰면 아이작 심을 종료할 수 있음
- 저 형태로 종료가 안되는 경우 컨테이너에 들어가서 프로세스를 찾아서 죽이면 됨
- IsaacLab 실행간에 디스플레이가 필요한 경우 `set -a && source docker/.env.scanbot && set +a`으로 환경변수를 로딩하면 `DISPLAY` 변수가 설정됨

# 프로젝트
- `scanbot` 폴더가 내가 주요하게 작업하는 공간임
- `scanbot/extension-manager`는 수정한 `extension`을 즉시 재로딩함 (파일을 수정하자마자 로그를 확인하면서 디버깅하면 됨)
- `scanbot/exec_bridge`를 통해서 필요한 코드 스니핏을 실행시켜볼 수 있음
- `docs` 폴더에는 `IsaacLab` 문서가 있음, 새로운 기능을 구현하거나 할 때 먼저 이곳에서 확인해보고 할 것
- `/mnt/ext_sda1/dev/isaac_lab_scanbot/scripts/environments/teleoperation/*collect3d*`에는 내가 만들고자 하는거의 지저분한 레거시 버전이 있음, 종종 이곳에서 기능을 마이그레이션해야 할 수도 있음
- 지금 `isaac-lab-scanbot` 컨테이너에 프로젝트의 `scanbot` 경로가 `/workspace/scanbot`에 마운트 되어있음
- `scanbot/AGENTS.md`에는 조금 더 시스템 또는 프로젝트 단위 지침이 적혀있음
- `README.md`에 전체 프로젝트의 간단한 설명과 개발 방향 등이 적혀 있음

# 컨테이너 제어
- 컨테이너 제어 스크립트 실행 전에 DISPLAY 환경 변수를 위에서 언급한 것처럼 설정해야 할 수도 있음
- `docker/.env.scanbot`에 DISPLAY 환경 변수가 정의되지 않은 경우 사용자에게 어떤 값을 사용할지 물어볼 것
- 컨테이너 삭제를 요구하는 경우, `stop` → 관련 `volume`도 찾아서 삭제가 필요
```bash
docker/container.sh stop scanbot
# 컨테이너 종료
docker/container.sh start scanbot --no-build
# 빌드 없이 컨테이너 재시작, 재부팅 후나 꼬인 경우 이 명령어로 컨테이너만 재시작
docker/container.sh build scanbot --target scanbot
# 스캔봇 이미지만 재빌드, 
# start 명령은 `--no-build` 없이 실행 시 빌드 + 실행을 같이합니다.
# build 명령은 이미지만 빌드합니다. --target 없이 빌드할 경우 scanbot 이미지의 부모 이미지인 base 이미지와 scanbot 이미지를 순차적으로 빌드합니다.
```

# 기타 당부 사항
- `is not None`, `hasattr`, `try-catch`와 같은 구문을 절대 남발하지 말 것 (필요한 곳에서는 써도 됨)
  - 필요한 테스트가 있으면 `exec_bridge` extension으로 테스트를 해볼 것
  - 항상 존재하는 프로퍼티나 값이라면 이런 Guard를 덕지덕지 붙일 필요가 없음
- `scanbot/scratch` 경로를 테스트 용으로 자유롭게 써도 됨
