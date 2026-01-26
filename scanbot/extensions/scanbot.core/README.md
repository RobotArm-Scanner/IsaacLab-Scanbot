# Scanbot Core

Scanbot 확장들의 공통 기반 확장입니다.

- `/World/default_camera`를 생성하고 기본 위치/회전을 설정합니다.
- 기본 `Viewport`에 해당 카메라를 활성 카메라로 설정합니다.
- `_camera`로 끝나는 카메라들 중 `free_wrist`를 제외하고 6개를 찾아(기본 카메라 제외) 카메라 이름으로 뷰포트를 만들고 오른쪽에 2열 그리드로 도킹합니다.

경로: `scanbot/extensions/scanbot.core`
