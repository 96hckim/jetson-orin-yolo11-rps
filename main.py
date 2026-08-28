"""
main.py
- TensorRPS 애플리케이션의 메인 이벤트 루프 및 실행 엔트리포인트.
- 저지연 카메라 캡처, TensorRT 제스처 탐지, 게임 모드 FSM 갱신, UI 렌더링을 동기화하여 처리.
"""

import sys
import time

import cv2
from configs.app_config import AppConfig
from core.detector import YOLOv11Detector
from game.modes.base import BaseGameMode
from game.modes.pve import PvEMode
from game.modes.pvp import PvPMode
from game.types import GameMode, GameState
from ui.renderer import GameUIRenderer


def create_camera_capture(
    device_id: int, width: int, height: int, buffer_size: int
) -> cv2.VideoCapture:
    """고속 저지연 V4L2 카메라 캡처 인스턴스 생성"""
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_id)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return cap


def main() -> None:
    config = AppConfig()

    print("⚡ TensorRT 비전 엔진 초기화 중...")
    detector = YOLOv11Detector(config.vision)
    renderer = GameUIRenderer(countdown_sec=config.game.countdown_seconds)

    cap = create_camera_capture(
        config.camera.device_id,
        config.camera.width,
        config.camera.height,
        config.camera.buffer_size,
    )

    if not cap.isOpened():
        print(f"❌ 카메라 디바이스({config.camera.device_id})를 열 수 없습니다.")
        sys.exit(1)

    # 게임 모드 Enum 키 기반 초기화
    modes: dict[GameMode, BaseGameMode] = {
        GameMode.PVP: PvPMode(config.game),
        GameMode.PVE_NORMAL: PvEMode(config.game, god_mode=False),
        GameMode.PVE_GOD: PvEMode(config.game, god_mode=True),
    }
    current_mode = GameMode.PVP

    print("🚀 TensorRPS 시스템 가동 시작!")
    print("   - [SPACE]: 라운드 시작")
    print("   - [1]: 2인 대전 모드 (PvP)")
    print("   - [2]: PC 일반 대전 모드 (PvE Normal)")
    print("   - [3]: PC 무적 모드 (PvE God Mode)")
    print("   - [R]: 스코어 리셋")
    print("   - [Q / ESC]: 프로그램 종료")

    prev_time = time.time()
    fps = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 프레임 수신 실패")
                break

            # 거울 모드(좌우 반전) 적용
            frame = cv2.flip(frame, 1)

            # 1. 딥러닝 추론 (손 위치 및 제스처 검출)
            detections = detector.detect(frame)

            # 2. 현재 활성화된 게임 모드 업데이트
            active_game = modes[current_mode]
            context = active_game.update(detections, frame.shape[1])

            # 3. FPS 연산 (지수이동평균 EMA 필터링)
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            curr_fps = 1.0 / max(dt, 1e-6)
            fps = 0.9 * fps + 0.1 * curr_fps if fps > 0 else curr_fps

            # 4. 종합 화면 렌더링
            renderer.render(
                frame,
                context,
                detections,
                current_mode,
                fps,
            )

            cv2.imshow("TensorRPS - Jetson Embedded Engine", frame)

            # 5. 키 이벤트 핸들링
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q or ESC
                break
            elif key == ord(" "):  # Space
                if context.state in (GameState.IDLE, GameState.ROUND_RESULT):
                    active_game.start_round()
            elif key == ord("1"):
                current_mode = GameMode.PVP
                modes[current_mode].reset_match()
                print("🔄 모드 전환: [PvP 2인 대전]")
            elif key == ord("2"):
                current_mode = GameMode.PVE_NORMAL
                modes[current_mode].reset_match()
                print("🔄 모드 전환: [PvE 일반 PC 대전]")
            elif key == ord("3"):
                current_mode = GameMode.PVE_GOD
                modes[current_mode].reset_match()
                print("🔄 모드 전환: [PvE 얌체 God Mode (무적 PC)]")
            elif key == ord("r"):
                modes[current_mode].reset_match()
                print("🔄 현재 매치 스코어 초기화 완료")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 TensorRPS 프로그램이 정상 종료되었습니다.")


if __name__ == "__main__":
    main()
