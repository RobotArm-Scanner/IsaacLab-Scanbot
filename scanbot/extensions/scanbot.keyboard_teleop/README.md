# Scanbot Keyboard Teleop

WASDQE 키 입력으로 Piper를 조작하는 간단한 teleop 확장입니다. 기본 런처에서 등록한 `env`와 `app`을 `scanbot_context`를 통해 가져와 매 프레임 `env.step`을 호출합니다.

- 움직임: `W/S`(x), `A/D`(y), `Q/E`(z). 나머지 축은 0으로 둡니다.
- 리셋: `R` 키를 누르면 기본 자세로 돌아가며, 이후 이동 키 입력 전까지 기본 자세를 유지합니다.
- 액션 스케일은 기본 0.5입니다(코드 상수 변경 가능).
- 키보드 상태는 Omniverse input 인터페이스로 직접 구독합니다. 모듈 존재 여부를 확인하기 위한 try/except는 사용하지 않습니다.

경로: `scanbot/extensions/scanbot.keyboard_teleop`

사용:
1) `--ext-folder scanbot/extensions`를 포함해 Isaac Lab을 실행하고 `scanbot.extension_manager` 또는 직접 `scanbot.keyboard_teleop`를 enable 합니다.
2) `basic_launcher.py`를 통해 env를 띄운 뒤, 창 포커스를 유지한 채 WASDQE로 조작하면 됩니다. 입력이 없을 때는 0 액션으로 프레임을 진행합니다.***
