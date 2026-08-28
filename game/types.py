"""
game/types.py
- 프로젝트 전역에서 공유하는 핵심 데이터 타입, 열거형(Enum) 및 불변 구조체 정의.
- Gesture(제스처), GameMode(모드), GameState(FSM 단계), GameResult(판정), Detection(손 검출) 정의.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum, auto


class GameMode(Enum):
    """게임 운영 모드 열거형"""

    PVP = auto()
    PVE_NORMAL = auto()
    PVE_GOD = auto()

    @property
    def display_name(self) -> str:
        """HUD 및 터미널 출력용 모드 명칭"""
        names = {
            GameMode.PVP: "PVP",
            GameMode.PVE_NORMAL: "PVE_NORMAL",
            GameMode.PVE_GOD: "PVE_GOD",
        }
        return names[self]

    @property
    def is_pvp(self) -> bool:
        """2인 대전 모드 여부 플래그"""
        return self == GameMode.PVP


class Gesture(IntEnum):
    """YOLO 모델 학습 클래스 순서(0, 1, 2)와 1:1 매핑되는 제스처 열거형"""

    PAPER = 0
    ROCK = 1
    SCISSORS = 2
    NONE = -1  # 손 미검출 또는 대기 상태

    @property
    def display_name(self) -> str:
        names = {
            Gesture.PAPER: "PAPER (보)",
            Gesture.ROCK: "ROCK (바위)",
            Gesture.SCISSORS: "SCISSORS (가위)",
            Gesture.NONE: "READY...",
        }
        return names.get(self, "UNKNOWN")


class GameState(Enum):
    """게임 수명 주기 FSM 상태 열거형"""

    IDLE = auto()  # 대기 상태 (라운드 시작 전)
    COUNTDOWN = auto()  # 3, 2, 1 카운트다운
    JUDGING = auto()  # 다수결 프레임 수집 및 판정
    ROUND_RESULT = auto()  # 라운드 승패 결과 출력
    MATCH_OVER = auto()  # 최종 세트 종료


class GameResult(Enum):
    """라운드 승패 판정 결과 열거형"""

    PLAYER1_WIN = "PLAYER 1 WIN"
    PLAYER2_WIN = "PLAYER 2 WIN"
    DRAW = "DRAW"
    INVALID = "NO HAND DETECTED"


@dataclass(slots=True, frozen=True)
class Detection:
    """단일 손 검출 결과 구조체 (불변 객체)"""

    gesture: Gesture
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)

    @property
    def center_x(self) -> float:
        """PvP 좌/우 화면 분할 판별용 바운딩 박스 중심 X 좌표"""
        return (self.bbox[0] + self.bbox[2]) / 2.0
