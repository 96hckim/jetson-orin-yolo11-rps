# 불변(Immutable) 설정 데이터클래스
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """비전 엔진 및 TensorRT 관련 설정"""

    engine_path: Path = (
        Path(__file__).parent.parent / "models" / "rps_yolo11n_custom_640.engine"
    )
    input_size: tuple[int, int] = (640, 640)
    conf_threshold: float = 0.50
    iou_threshold: float = 0.45


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """카메라 디바이스 및 비디오 스트림 설정"""

    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass(frozen=True, slots=True)
class GamePlayConfig:
    """게임 상태 머신 및 룰 설정"""

    countdown_seconds: float = 3.0
    result_display_seconds: float = 2.5
    voting_window_frames: int = 10  # 다수결 판정 시 수집할 프레임 수
    target_score: int = 3  # 세트 승리 점수 (3판 2선승제 등)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """전체 애플리케이션 통합 설정"""

    vision: VisionConfig = VisionConfig()
    camera: CameraConfig = CameraConfig()
    game: GamePlayConfig = GamePlayConfig()
