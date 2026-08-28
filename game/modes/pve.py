# 2인 대전 모드 (좌우 화면 분할)
import random

from configs.app_config import GamePlayConfig

from game.modes.base import BaseGameMode
from game.rules import get_counter_gesture
from game.types import Detection, GameState, Gesture


class PvEMode(BaseGameMode):
    """플레이어 vs AI 대전 모드"""

    def __init__(self, config: GamePlayConfig, god_mode: bool = False) -> None:
        super().__init__(config)
        self.god_mode = god_mode
        self._pc_fixed_gesture: Gesture = Gesture.NONE

    def start_round(self) -> None:
        super().start_round()
        # 일반 모드일 경우 카운트다운 시작 시점에 PC 패를 미리 결정
        if not self.god_mode:
            self._pc_fixed_gesture = random.choice(
                [Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS]
            )

    def extract_player_gestures(
        self,
        detections: list[Detection],
        frame_width: int,
    ) -> tuple[Gesture, Gesture]:
        # 화면 내에서 가장 신뢰도 높은 손을 플레이어의 패로 선택
        p1_gesture = Gesture.NONE
        if detections:
            best_det = max(detections, key=lambda d: d.confidence)
            p1_gesture = best_det.gesture

        # 상태에 따른 AI 패 결정
        if self.state_machine.context.state == GameState.COUNTDOWN:
            # 카운트다운 중에는 룰렛처럼 랜덤하게 섞어서 출력
            pc_gesture = random.choice([Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS])
        elif self.state_machine.context.state == GameState.JUDGING:
            if self.god_mode:
                # 젯슨 초저지연 연계: 유저 패를 실시간으로 읽어 무조건 이기는 패 선택
                pc_gesture = (
                    get_counter_gesture(p1_gesture)
                    if p1_gesture != Gesture.NONE
                    else Gesture.ROCK
                )
            else:
                pc_gesture = self._pc_fixed_gesture
        else:
            pc_gesture = self._pc_fixed_gesture

        return p1_gesture, pc_gesture
