from abc import ABC, abstractmethod

from configs.app_config import GamePlayConfig

from game.state_machine import GameContext, GameStateMachine
from game.types import Detection, Gesture


class BaseGameMode(ABC):
    """모든 게임 모드의 기본 추상 클래스 (Strategy Pattern)"""

    def __init__(self, config: GamePlayConfig) -> None:
        self.config = config
        self.state_machine = GameStateMachine(config)

    @property
    def context(self) -> GameContext:
        return self.state_machine.context

    def start_round(self) -> None:
        self.state_machine.start_round()

    def reset_match(self) -> None:
        self.state_machine.reset_match()

    @abstractmethod
    def extract_player_gestures(
        self,
        detections: list[Detection],
        frame_width: int,
    ) -> tuple[Gesture, Gesture]:
        """검출된 객체 목록으로부터 P1, P2의 실시간 제스처를 분리 및 결정하는 추상 메서드"""

    def update(
        self,
        detections: list[Detection],
        frame_width: int,
    ) -> GameContext:
        """프레임별 제스처 추출 후 상태 머신 갱신"""
        p1_gesture, p2_gesture = self.extract_player_gestures(detections, frame_width)
        return self.state_machine.update(p1_gesture, p2_gesture)
