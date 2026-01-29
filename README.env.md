# Scanbot 환경 설정

Scanbot 실행에 관련된 환경 식별자(task id)를 정리합니다.

## Scanbot 환경 식별자 (Gym task id)

`scanbot/bin/scanbot.sh`는 `SCANBOT_TASK` 환경변수로 task id를 선택합니다. 기본값은 `e2`입니다.

- `e2`: 기본 스캔 환경 (t1 리소스 기본). `scanbot/scripts/cfg/scanbot_e2_cfg.py:ScanbotE2Cfg`
- `e2.t1`: `e2`와 동일한 별칭(alias). `ScanbotE2Cfg`
- `e2.t1.rt1`: t1(더 크게 열린 mouth) + Y축 20도 회전. `ScanbotE2T1RT1Cfg`
- `e2.t2.rt1`: t2(`9000.usd`) + 오프셋/스케일 적용. `ScanbotE2T2RT1Cfg`
- `e2.t3ds`: Teeth3DS(OBJ→USD) 데이터셋 사용 (dataset_id=`A9TECAGP`). `ScanbotE2T3DSCfg`
- `e2.t3ds.rl`: `e2.t3ds` 기반 RL 설정(+ `rsl_rl` runner). `ScanbotE2RLT3DSCfg`

### 예시

```bash
# 컨테이너 내부
SCANBOT_TASK=e2.t3ds scanbot/bin/scanbot.sh
```

### 정의 위치

- task 등록: `scanbot/scripts/scanbot_task.py`
- cfg 구현: `scanbot/scripts/cfg/scanbot_e2_cfg.py`

