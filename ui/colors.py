"""
ui/colors.py
- OpenCV GUI 렌더링 전용 BGR 색상 팔레트 및 세련된 네온 테마
"""

from game.types import Gesture

# 배경 및 섀도우
COLOR_BG_DARK: tuple[int, int, int] = (15, 17, 23)  # 딥 다크 네이비 (#17110F)
COLOR_CARD_BG: tuple[int, int, int] = (25, 28, 36)  # 글래스모피즘 카드 배경
COLOR_BORDER: tuple[int, int, int] = (60, 65, 80)  # 카드 테두리 그레이

# 기본 텍스트 및 악센트
COLOR_WHITE: tuple[int, int, int] = (245, 245, 245)
COLOR_GRAY: tuple[int, int, int] = (130, 135, 150)
COLOR_CYAN: tuple[int, int, int] = (240, 195, 0)  # 네온 시안 (P1 악센트)
COLOR_AMBER: tuple[int, int, int] = (0, 185, 255)  # 일렉트릭 골드/오렌지
COLOR_GREEN: tuple[int, int, int] = (90, 225, 80)  # 네온 에메랄드 (승리)
COLOR_RED: tuple[int, int, int] = (70, 70, 245)  # 네온 크림슨 (패배)
COLOR_BLUE: tuple[int, int, int] = (245, 140, 50)  # 스카이 블루 (무승부)

# 제스처별 시각화 색상
GESTURE_COLORS: dict[Gesture, tuple[int, int, int]] = {
    Gesture.PAPER: (240, 195, 0),  # 시안
    Gesture.ROCK: (90, 225, 80),  # 에메랄드
    Gesture.SCISSORS: (70, 70, 245),  # 크림슨
    Gesture.NONE: (110, 115, 130),
}
