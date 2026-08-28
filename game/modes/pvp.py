"""
game/modes/pvp.py
- 2인 로컬 대전 모드 (PvP).
- 단일 카메라 프레임을 좌/우 화면으로 2분할하여 각 영역의 최고 신뢰도 제스처를 독립 판별.
"""

from game.modes.base import BaseGameMode
from game.types import Detection, Gesture


class PvPMode(BaseGameMode):
    """2인 오프라인 대전 모드 (화면 좌/우 영역 분할)"""

    def extract_player_gestures(
        self,
        detections: list[Detection],
        frame_width: int,
    ) -> tuple[Gesture, Gesture]:
        if not detections:
            return Gesture.NONE, Gesture.NONE

        half_x = frame_width / 2.0
        p1_candidates: list[Detection] = []
        p2_candidates: list[Detection] = []

        for det in detections:
            if det.center_x < half_x:
                p1_candidates.append(det)
            else:
                p2_candidates.append(det)

        # 각 영역에서 Confidence가 가장 높은 검출 결과를 채택
        p1_gesture = Gesture.NONE
        if p1_candidates:
            p1_best = max(p1_candidates, key=lambda d: d.confidence)
            p1_gesture = p1_best.gesture

        p2_gesture = Gesture.NONE
        if p2_candidates:
            p2_best = max(p2_candidates, key=lambda d: d.confidence)
            p2_gesture = p2_best.gesture

        return p1_gesture, p2_gesture
