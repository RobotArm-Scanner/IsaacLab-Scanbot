# IsaacLab-Scanbot: 스캔봇 통합 시뮬레이션 환경

강화학습 + 모방학습 + ROS2 전부를 지원하는 통합 시뮬레이션 환경을 구현합니다.

```bash
scanbot
 - bin: 시뮬레이터를 켜기 위한 시동 스크립트를 이곳에 정의합니다.
 - extensions: 시뮬레이터에 통합되어야 할 추가기능을 이곳에 정의합니다.
 - logs: 로깅용 디렉토리입니다. 자유롭게 활용해도 좋습니다. (내용물은 커밋 금지)
 - resources: usd 파일 등의 시뮬레이션 리소스가 관리되는 폴더입니다.
 - ros2: scanbot의 ros2 통합 소스들이 이곳에 위치합니다.
 - scratch: 개발간 필요한 스크래치 디렉토리입니다. 중요한 잡동사니 파일이 아니면 이 폴더에다 새로운 파일을 넣고 커밋하지 마세요 
 - scripts: Scanbot 시뮬레이터의 코어/환경 소스가 정의되는 곳입니다. 이곳의 소스코드는 조금 더 엄격하게 작성되어야 합니다.
 - snippets: exec_bridge extension으로 실행 가능한 스니핏들을 이곳에 저장합니다.
```

## Extensions
- [scanbot.extension_manager](scanbot/extensions/scanbot.extension_manager/README.md): scanbot.* 확장 자동 enable + 파일 변경 시 자동 reload
- [scanbot.core](scanbot/extensions/scanbot.core/README.md): 기본 카메라/뷰포트 세팅과 뷰포트 레이아웃 구성 (카메라 설정이 아니라 카메라 UI)
- [scanbot.common](scanbot/extensions/scanbot.common/README.md): 확장 간 공용 유틸리티 모음
- [scanbot.exec_bridge](scanbot/extensions/scanbot.exec_bridge/README.md): 실행 중 코드 스니핏을 HTTP로 실행하는 브리지
- [scanbot.keyboard_teleop](scanbot/extensions/scanbot.keyboard_teleop/README.md): WASDQE 키보드 텔레옵
- [scanbot.random_pose](scanbot/extensions/scanbot.random_pose/README.md): 랜덤 액션 1회 실행 버튼 UI
- [scanbot.ros2_manager](scanbot/extensions/scanbot.ros2_manager/README.md): ROS2 액션/토픽/서비스 브리지
- [scanbot.simple_gui](scanbot/extensions/scanbot.simple_gui/README.md): 최소 예제 GUI (슬라이더 + 버튼)

## 개발 중요 원칙
1. 컨테이너 내부의 설치 요소들은 환경 통일을 위해서, `Dockerfile`에 관련 구성이 정의되어야 합니다. 
 - 환경에 포함되지 않은 의존성/리소스를 본인 컨테이너에만 설치하고, 소스에서 참고하는 형태가 되어서는 안됩니다.
 - 번거롭더라도 컨테이너를 빌드해서 환경이 세팅이 되는지 확인 후 커밋해주세요.
2. 현재 프로젝트는 소스 코드 품질을 위해서 다소 번거롭더라도 PR 검토를 할 예정입니다.
3. 나중에 쓰지 못하는 구조로 코드를 짜는 것을 지양해주세요. (더 큰 비용으로 돌아옵니다)
4. 다른 사람이 직관적으로 이해가 가능한 디렉토리와 네이밍 구조를 사용하세요.
5. 각각의 디렉토리에 필요한 경우 `README.md`로 문서화를 하는 것을 적극 권장합니다.
6. main에 직접 커밋하는 것은 `hotfix`가 아닌 경우 지양해주세요.
7. 소스코드가 300~500줄을 넘어가는 경우, 대부분 잘못된 구조인 경우가 많습니다. 리팩토링을 적용해서 커밋해주세요.
8. `Single Source of Truth(같은 정보는 한 곳에서만 정의)`, `Single Responsibility Principle(한 모듈은 한 책임만)` 원칙을 지켜주세요.
9. `source/*`에는 메인스트림 레포지토리의 소스가 들어가있습니다. 이곳의 소스코드는 직접 수정하거나, 우리의 소스를 같이 두는 것을 지양해주세요. (관리가 어려워집니다)
10. 환경과 분리되어야 할 기능은 `Extension`으로 작성합니다. `Extension`에서는 `scanbot/scripts/scanbot_context.py`를 통해 환경을 간접적으로 제어할 수 있습니다.
11. `Extension`단에서 공유되어야 할 코드는 `scanbot.common`에 작성합니다. `Extension` 및 전체 프로젝트 코드에서 공유되어야 할 코드는 `scripts/utilities/*`에 작성하는 것이 좋습니다. 


## 브랜치 관리
간소화된 `git flow` 방법을 사용합니다.
- `feature/*` 브랜치에 새 기능을 작성합니다.
- 기능 개발이 완료되면 `main`에 PR로 병합합니다. (`gh` 명령어를 이용해서 `AI Agent`에게 PR을 부탁할 수 있습니다.)
- 개발이 완료된 브랜치는 가급적 삭제 처리해주세요.
- 보존해야 할 브랜치의 경우 `archive/*`로 남겨주세요.

## 커밋 메시지 컨벤션

이 프로젝트는 현재 Conventional Commits를 강제하지 않고, 최근 커밋 기준으로 아래 스타일을 사용합니다.

- 기본: `<Verb> <What>` 형태의 영어 명령형(Imperative) 1줄 (첫 글자 대문자, 마침표 없이)
- 스코프가 필요한 경우: `<Verb> <scope>: <What>` (예: `Fix extension_manager: ignore noisy FS events`)
- 파일/대상을 보조로 표기할 때는 괄호를 사용할 수 있습니다. (예: `Minor fix (scanbot.sh)`)
- 자주 쓰는 Verb 예시: `Add`, `Fix`, `Update`, `Move`, `Refactor`, `Rename`, `Simplify`, `Drop`, `Replace`, `Implement`, `Improve`

## 권장 개발 환경 세팅
### 필수 점검 사항
- `/etc/sudoers`를 편집하여 `USERNAME ALL=(ALL:ALL) NOPASSWD: ALL`을 추가해서 호스트를 자율제어 할 수 있게 합니다.
- `docker/.env.scanbot`의 `PROJECT_SUFFIX`를 `main`, `dev1`등으로 지정하면, 동일한 다른 `scanbot` 프로젝트와 컨테이너/이미지 충돌을 피할 수 있습니다. 

### 권장 점검 사항
- `tmux` 패키지 설치
- `VNC`는 호스트에 설치하여 사용합니다. (`TigerVNC`)
- `AI Agent`는 컨테이너 내부가 아니라 호스트에서 실행합니다.

### AI Agent 권장 설정
`codex --yolo`등의 자율 제어 모드로 실행을 권장

## tmux-mcp
```bash
npm install -g tmux-mcp
```

```toml
[mcp_servers.tmux]
command = "npx"
args = ["-y", "--no-install", "tmux-mcp"]
```
## 웹 검색 요청 허용
```toml
[features]
web_search_request = true
```

## AI Agent 기반 개발 예시 메시지
- 권장 환경 세팅이 되었는지 점검해 줘
- 아이작랩을 껐다가 켜줘
- `exec_bridge`로 현재 `wrist_camera`를 확인하고 판단해 줘
- `simple_gui` extension을 기반으로 새 extension을 만들어 줘
- `feature/test`에 ~ 기능을 만들고 `main`에 설명과 함께 PR해 줘 (`gh` 명령어가 설치되어 있고, 로그인된 경우)
- 지금까지 개발한 내용들 현재 브랜치에 커밋 및 푸시해 줘
