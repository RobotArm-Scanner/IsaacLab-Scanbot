# Scanbot Extension Manager

Scanbot 전용 확장 매니저입니다.

- 같은 디렉토리의 `scanbot.*` 확장을 자동 enable
- 파일 변경을 감지해서 해당 확장을 자동 reload (disable -> enable)
- `extension.toml`의 `scanbot.disabled=true`면 제외
- `HEADLESS=1` 또는 `SCANBOT_DISABLE_UI_EXTENSIONS=1`이면 UI 관련 확장 비활성화

경로: `scanbot/extensions/scanbot.extension_manager`
