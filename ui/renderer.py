# HUD, 스코어보드, 라벨 드로잉 분리
import time
from typing import ClassVar

import cv2
import numpy as np
from game.state_machine import GameContext
from game.types import Detection, GameResult, GameState, Gesture


class GameUIRenderer:
    """게임 HUD, 바운딩 박스 및 상태 오버레이 렌더러"""

    # BGR 색상 팔레트 상수 (불변 튜플)
    COLOR_BG_DARK: tuple[int, int, int] = (20, 20, 20)
    COLOR_WHITE: tuple[int, int, int] = (255, 255, 255)
    COLOR_GRAY: tuple[int, int, int] = (120, 120, 120)
    COLOR_GREEN: tuple[int, int, int] = (0, 220, 0)
    COLOR_BLUE: tuple[int, int, int] = (255, 160, 0)
    COLOR_RED: tuple[int, int, int] = (0, 0, 240)
    COLOR_YELLOW: tuple[int, int, int] = (0, 220, 255)

    # ClassVar를 통한 가변 객체 경고 해제 및 명시적 타입 정의
    GESTURE_COLORS: ClassVar[dict[Gesture, tuple[int, int, int]]] = {
        Gesture.PAPER: (255, 180, 50),
        Gesture.ROCK: (50, 205, 50),
        Gesture.SCISSORS: (50, 50, 255),
        Gesture.NONE: (150, 150, 150),
    }

    def __init__(self, countdown_sec: float = 3.0) -> None:
        self.countdown_sec = countdown_sec

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> None:
        """검출된 손 바운딩 박스 및 신뢰도 태그 렌더링"""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.GESTURE_COLORS.get(det.gesture, self.COLOR_WHITE)

            # 바운딩 박스 렌더링
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 상단 라벨 배경 박스
            label = f"{det.gesture.name} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - baseline - 6),
                (x1 + tw + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def draw_hud(
        self,
        frame: np.ndarray,
        context: GameContext,
        mode_name: str,
        fps: float,
        is_pvp: bool,
    ) -> None:
        """상단 스코어보드 및 매치 정보 렌더링"""
        h, w = frame.shape[:2]

        # 상단 반투명 헤더 바
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 모드 및 FPS 정보
        cv2.putText(
            frame,
            f"MODE: {mode_name}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.COLOR_YELLOW,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        # 중앙 스코어보드 (P1 vs P2/PC)
        p1_label = "PLAYER 1"
        p2_label = "PLAYER 2" if is_pvp else "PC (AI)"
        score_text = (
            f"{p1_label} [ {context.p1_score}  :  {context.p2_score} ] {p2_label}"
        )
        (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            frame,
            score_text,
            ((w - tw) // 2, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

        # PvP 모드일 때 화면 중앙 분할 가이드 라인
        if is_pvp:
            half_x = w // 2
            cv2.line(
                frame,
                (half_x, 60),
                (half_x, h),
                self.COLOR_GRAY,
                1,
                cv2.LINE_AA,
            )

    def draw_state_overlays(
        self, frame: np.ndarray, context: GameContext, is_pvp: bool
    ) -> None:
        """상태 머신 단계별 대형 텍스트 및 안내 오버레이"""
        h, w = frame.shape[:2]
        center_y = h // 2  # 사용되지 않던 center_x 제거

        match context.state:
            case GameState.IDLE:
                help_text = "Press [SPACE] to Start Round | '1':PvP '2':PvE '3':GodMode 'R':Reset"
                (tw, _), _ = cv2.getTextSize(
                    help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.putText(
                    frame,
                    help_text,
                    ((w - tw) // 2, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.COLOR_YELLOW,
                    1,
                    cv2.LINE_AA,
                )

            case GameState.COUNTDOWN:
                elapsed = time.time() - context.state_timer
                remain = max(0.0, self.countdown_sec - elapsed)
                count_str = str(int(np.ceil(remain))) if remain > 0.5 else "GO!"
                color = self.COLOR_YELLOW if remain > 0.5 else self.COLOR_GREEN

                (tw, th), _ = cv2.getTextSize(
                    count_str, cv2.FONT_HERSHEY_DUPLEX, 3.0, 5
                )
                cv2.putText(
                    frame,
                    count_str,
                    ((w - tw) // 2, center_y + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    3.0,
                    color,
                    5,
                    cv2.LINE_AA,
                )

            case GameState.JUDGING:
                judging_text = "JUDGING..."
                (tw, th), _ = cv2.getTextSize(
                    judging_text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3
                )
                cv2.putText(
                    frame,
                    judging_text,
                    ((w - tw) // 2, center_y + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.3,
                    self.COLOR_BLUE,
                    3,
                    cv2.LINE_AA,
                )

            case GameState.ROUND_RESULT:
                res_color = self.COLOR_GRAY
                if context.last_result == GameResult.PLAYER1_WIN:
                    res_color = self.COLOR_GREEN
                elif context.last_result == GameResult.PLAYER2_WIN:
                    res_color = self.COLOR_RED
                elif context.last_result == GameResult.DRAW:
                    res_color = self.COLOR_BLUE

                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (0, center_y - 60),
                    (w, center_y + 60),
                    self.COLOR_BG_DARK,
                    -1,
                )
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                result_text = context.last_result.value
                (tw, _), _ = cv2.getTextSize(
                    result_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3
                )
                cv2.putText(
                    frame,
                    result_text,
                    ((w - tw) // 2, center_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.2,
                    res_color,
                    3,
                    cv2.LINE_AA,
                )

                hands_info = (
                    f"P1: {context.p1_final_gesture.name}   vs   "
                    f"{'P2' if is_pvp else 'PC'}: {context.p2_final_gesture.name}"
                )
                (tw2, _), _ = cv2.getTextSize(
                    hands_info, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
                )
                cv2.putText(
                    frame,
                    hands_info,
                    ((w - tw2) // 2, center_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    self.COLOR_WHITE,
                    2,
                    cv2.LINE_AA,
                )

            case GameState.MATCH_OVER:
                winner = (
                    "PLAYER 1"
                    if context.p1_score > context.p2_score
                    else ("PLAYER 2" if is_pvp else "PC (AI)")
                )
                over_text = f"MATCH WINNER: {winner}!"
                (tw, _), _ = cv2.getTextSize(over_text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
                cv2.putText(
                    frame,
                    over_text,
                    ((w - tw) // 2, center_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.1,
                    self.COLOR_YELLOW,
                    3,
                    cv2.LINE_AA,
                )

                sub_text = "Press 'R' to Start a New Match"
                (tw2, _), _ = cv2.getTextSize(
                    sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.putText(
                    frame,
                    sub_text,
                    ((w - tw2) // 2, center_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.COLOR_WHITE,
                    1,
                    cv2.LINE_AA,
                )

    def render(
        self,
        frame: np.ndarray,
        context: GameContext,
        detections: list[Detection],
        mode_name: str,
        fps: float,
        is_pvp: bool,
    ) -> None:
        """전체 렌더링 파이프라인 통합 호출"""
        self.draw_detections(frame, detections)
        self.draw_hud(frame, context, mode_name, fps, is_pvp)
        self.draw_state_overlays(frame, context, is_pvp)
