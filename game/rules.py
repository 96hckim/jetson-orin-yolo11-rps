"""
game/rules.py
- 가위바위보 상성 판정 및 카운터 패 생성을 담당하는 순수 함수(Pure Logic) 모듈.
- 외부 상태 의존성 없이 입력 제스처 간의 승/패/무 판정 및 무적(God Mode) 반격 제스처 계산.
"""

from game.types import GameResult, Gesture

_WINNING_CONDITIONS: dict[Gesture, Gesture] = {
    Gesture.ROCK: Gesture.SCISSORS,
    Gesture.SCISSORS: Gesture.PAPER,
    Gesture.PAPER: Gesture.ROCK,
}

_COUNTER_GESTURES: dict[Gesture, Gesture] = {
    Gesture.ROCK: Gesture.PAPER,
    Gesture.PAPER: Gesture.SCISSORS,
    Gesture.SCISSORS: Gesture.ROCK,
}


def judge_round(p1_gesture: Gesture, p2_gesture: Gesture) -> GameResult:
    """두 제스처를 입력받아 승패를 반환하는 순수 판정 함수"""
    if p1_gesture == Gesture.NONE or p2_gesture == Gesture.NONE:
        return GameResult.INVALID

    if p1_gesture == p2_gesture:
        return GameResult.DRAW

    if _WINNING_CONDITIONS.get(p1_gesture) == p2_gesture:
        return GameResult.PLAYER1_WIN
    else:
        return GameResult.PLAYER2_WIN


def get_counter_gesture(user_gesture: Gesture) -> Gesture:
    """[God Mode용] 상대 패를 무조건 이기는 치트 패를 반환"""
    return _COUNTER_GESTURES.get(user_gesture, Gesture.ROCK)
