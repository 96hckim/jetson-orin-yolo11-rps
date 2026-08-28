# 가위바위보 판정 순수 함수 (Pure Logic)
from game.types import GameResult, Gesture


def judge_round(p1_gesture: Gesture, p2_gesture: Gesture) -> GameResult:
    """두 제스처를 입력받아 승패를 반환하는 순수 판정 함수"""
    if p1_gesture == Gesture.NONE or p2_gesture == Gesture.NONE:
        return GameResult.INVALID

    if p1_gesture == p2_gesture:
        return GameResult.DRAW

    # (Player 1 승리 조건: Key가 P1, Value가 P2일 때)
    winning_conditions = {
        Gesture.ROCK: Gesture.SCISSORS,
        Gesture.SCISSORS: Gesture.PAPER,
        Gesture.PAPER: Gesture.ROCK,
    }

    if winning_conditions.get(p1_gesture) == p2_gesture:
        return GameResult.PLAYER1_WIN

    return GameResult.PLAYER2_WIN


def get_counter_gesture(user_gesture: Gesture) -> Gesture:
    """[God Mode용] 상대 패를 무조건 이기는 치트 패를 반환"""
    counters = {
        Gesture.ROCK: Gesture.PAPER,
        Gesture.PAPER: Gesture.SCISSORS,
        Gesture.SCISSORS: Gesture.ROCK,
    }
    return counters.get(user_gesture, Gesture.ROCK)
