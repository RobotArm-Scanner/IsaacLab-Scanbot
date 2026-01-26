# IsaacLab-ScanBot: 스캔봇 통합 시뮬레이션 환경

강화학습 + 모방학습 + ROS2 전부를 지원하는 통합 시뮬레이션 환경을 구현합니다.

```bash
scanbot
 - bin: 시뮬레이터를 켜기 위한 시동 스크립트를 이곳에 정의합니다.
 - extensions: 시뮬레이터에 통합되어야 할 추가기능을 이곳에 정의합니다.
 - logs: 로깅용 디렉토리입니다. 자유롭게 활용해도 좋습니다. (내용물은 커밋 금지)
 - resources: usd 파일 등의 시뮬레이션 리소스가 관리되는 폴더입니다.
 - ros2: scanbot의 ros2 통합 소스들이 이곳에 위치합니다.
 - scratch: 개발간 필요한 스크래치 디렉토리입니다. 중요한 잡동사니 파일이 아니면 이 폴더에다 새로운 파일을 넣고 커밋하지 마세요 
 - scripts: ScanBot 시뮬레이터의 코어/환경 소스가 정의되는 곳입니다. 이곳의 소스는 조금 더 엄격하게 작성되어야 합니다.
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
1. 컨테이너 내부의 설치 요소들은 환경 통일을 위해서, Dockerfile에 관련 구성이 정의되어야 합니다. 
 - 환경에 포함되지 않은 의존성/리소스를 본인 컨테이너에만 설치하고, 소스에서 참고하는 형태가 되어서는 안됩니다.
 - 번거롭더라도 컨테이너를 빌드해서 환경이 세팅이 되는지 확인 후 커밋해주세요.
2. 현재 프로젝트는 소스 코드 품질을 위해서 다소 번거롭더라도 PR 검토를 할 예정입니다.
3. 나중에 쓰지 못하는 구조로 코드를 짜는 것을 지양해주세요. (더 큰 비용으로 돌아옵니다)
4. 다른 사람이 직관적으로 이해가 가능한 디렉토리와 네이밍 구조를 사용하세요.
5. 각각의 디렉토리에 필요한 경우 README.md로 문서화를 하는 것을 적극 권장합니다.
6. main에 직접 커밋하는 것은 핫픽스를 제외한 경우 불허합니다.
7. 소스코드가 300~500줄을 넘어가는 경우, 대부분 잘못된 구조인 경우가 많습니다. 리팩토링을 적용해서 커밋해주세요.

`source/*`에는 메인스트림 레포지토리의 소스가 들어가있습니다. 이곳의 소스코드는 직접 수정하거나, 우리의 소스를 같이 두는 것을 지양해주세요. (관리가 어려워집니다)
