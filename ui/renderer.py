# ui/renderer.py
import time
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from game.state_machine import GameContext
from game.types import Detection, GameResult, GameState, Gesture


class GameUIRenderer:
    """게임 HUD, 바운딩 박스 및 최적화된 그래픽 오버레이 렌더러"""

    # BGR 색상 팔레트 상수
    COLOR_BG_DARK: tuple[int, int, int] = (20, 20, 20)
    COLOR_WHITE: tuple[int, int, int] = (255, 255, 255)
    COLOR_GRAY: tuple[int, int, int] = (140, 140, 140)
    COLOR_GREEN: tuple[int, int, int] = (0, 230, 0)
    COLOR_BLUE: tuple[int, int, int] = (255, 170, 0)
    COLOR_RED: tuple[int, int, int] = (0, 0, 245)
    COLOR_YELLOW: tuple[int, int, int] = (0, 220, 255)

    GESTURE_COLORS: ClassVar[dict[Gesture, tuple[int, int, int]]] = {
        Gesture.PAPER: (255, 180, 50),
        Gesture.ROCK: (50, 205, 50),
        Gesture.SCISSORS: (50, 50, 255),
        Gesture.NONE: (120, 120, 120),
    }

    def __init__(
        self, countdown_sec: float = 3.0, assets_dir: Path | str | None = None
    ) -> None:
        self.countdown_sec = countdown_sec
        self.assets_dir = (
            Path(assets_dir) if assets_dir else Path(__file__).parent.parent / "assets"
        )
        self.icons: dict[Gesture, np.ndarray | None] = self._load_icons()

    def _load_icons(self) -> dict[Gesture, np.ndarray | None]:
        """assets 디렉터리에서 투명 PNG 아이콘 로드 (BGRA)"""
        mapping = {
            Gesture.ROCK: "rock.png",
            Gesture.PAPER: "paper.png",
            Gesture.SCISSORS: "scissors.png",
        }
        loaded = {}
        for gesture, filename in mapping.items():
            path = self.assets_dir / filename
            if path.exists():
                loaded[gesture] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            else:
                loaded[gesture] = None
        loaded[Gesture.NONE] = None
        return loaded

    # --- 렌더링 공통 헬퍼 메서드 ---
    @staticmethod
    def _draw_centered_text(
        frame: np.ndarray,
        text: str,
        center_x: int,
        y: int,
        font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
        scale: float = 1.0,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> None:
        """가로 중심(center_x) 기준으로 텍스트 완벽 중앙 정렬"""
        (tw, _), _ = cv2.getTextSize(text, font_face, scale, thickness)
        cv2.putText(
            frame,
            text,
            (center_x - tw // 2, y),
            font_face,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_overlay_rect(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        alpha: float = 0.7,
    ) -> None:
        """지정 영역에 알파 블렌딩 반투명 사각형 오버레이 적용"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def _overlay_icon(
        self,
        frame: np.ndarray,
        gesture: Gesture,
        center_x: int,
        center_y: int,
        size: int = 70,
    ) -> None:
        """중심 좌표(center_x, center_y) 기준으로 아이콘/플레이스홀더 렌더링"""
        half_sz = size // 2
        x = center_x - half_sz
        y = center_y - half_sz
        h, w = frame.shape[:2]

        if x < 0 or y < 0 or x + size > w or y + size > h:
            return

        icon = self.icons.get(gesture)
        if icon is not None:
            resized = cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)
            if resized.shape[2] == 4:  # RGBA 알파 채널 합성
                alpha = resized[:, :, 3] / 255.0
                for c in range(3):
                    frame[y : y + size, x : x + size, c] = (
                        alpha * resized[:, :, c]
                        + (1.0 - alpha) * frame[y : y + size, x : x + size, c]
                    )
                return
            frame[y : y + size, x : x + size] = resized[:, :, :3]
            return

        # 손이 없거나(Gesture.NONE) 에셋이 없을 때의 세련된 대체 박스
        color = self.GESTURE_COLORS.get(gesture, self.COLOR_GRAY)
        cv2.rectangle(frame, (x, y), (x + size, y + size), color, 2)
        label = "?" if gesture == Gesture.NONE else gesture.name[:3]
        self._draw_centered_text(
            frame,
            label,
            center_x,
            center_y + 6,
            font_face=cv2.FONT_HERSHEY_DUPLEX,
            scale=0.7,
            color=color,
            thickness=2,
        )

    # --- 개별 UI 컴포넌트 렌더링 ---
    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> None:
        """검출 바운딩 박스 및 라벨 렌더링"""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.GESTURE_COLORS.get(det.gesture, self.COLOR_WHITE)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det.gesture.name} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
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
        """상단 헤더 HUD 및 스코어보드 (텍스트 겹침 방지 레이아웃)"""
        h, w = frame.shape[:2]

        # 1. 상단 배경 바
        self._draw_overlay_rect(frame, 0, 0, w, 55, self.COLOR_BG_DARK, alpha=0.75)

        # 2. 좌측: 모드 & FPS (폰트 스케일 및 줄바꿈 최적화)
        cv2.putText(
            frame,
            f"MODE: {mode_name}",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.COLOR_YELLOW,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        # 3. 중앙: 스코어보드 (가로 정렬 최적화)
        p1_tag = "P1"
        p2_tag = "P2" if is_pvp else "PC"
        score_text = f"{p1_tag} [ {context.p1_score} : {context.p2_score} ] {p2_tag}"
        self._draw_centered_text(
            frame,
            score_text,
            center_x=w // 2,
            y=36,
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            scale=0.7,
            color=self.COLOR_WHITE,
            thickness=2,
        )

        # 4. PvP 분할선 또는 PvE AI 상태 카드
        if is_pvp:
            cv2.line(frame, (w // 2, 55), (w // 2, h), self.COLOR_GRAY, 1, cv2.LINE_AA)
        else:
            self._draw_pve_ai_card(frame, context)

    def _draw_pve_ai_card(self, frame: np.ndarray, context: GameContext) -> None:
        """우측 상단 AI 카드 (중앙 정렬 완벽 보정)"""
        w = frame.shape[1]
        card_w, card_h = 120, 115
        card_x, card_y = w - card_w - 12, 65

        # 반투명 카드 배경 및 테두리
        self._draw_overlay_rect(
            frame,
            card_x,
            card_y,
            card_x + card_w,
            card_y + card_h,
            self.COLOR_BG_DARK,
            alpha=0.65,
        )
        cv2.rectangle(
            frame,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            self.COLOR_GRAY,
            1,
        )

        # 카드 타이틀 중앙 정렬
        card_center_x = card_x + card_w // 2
        self._draw_centered_text(
            frame,
            "[ PC CHOICE ]",
            center_x=card_center_x,
            y=card_y + 20,
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            scale=0.42,
            color=self.COLOR_YELLOW,
            thickness=1,
        )

        # 상태별 AI 패 결정
        display_gesture = Gesture.NONE
        if context.state == GameState.ROUND_RESULT:
            display_gesture = context.p2_final_gesture
        elif (
            context.state in (GameState.COUNTDOWN, GameState.JUDGING)
            and context.p2_vote_buffer
        ):
            display_gesture = context.p2_vote_buffer[-1]

        # 내부 내용물 (아이콘 또는 '???') 중앙 배치
        content_center_y = card_y + 25 + (card_h - 25) // 2
        if display_gesture != Gesture.NONE:
            self._overlay_icon(
                frame,
                display_gesture,
                center_x=card_center_x,
                center_y=content_center_y,
                size=65,
            )
        else:
            self._draw_centered_text(
                frame,
                "???",
                center_x=card_center_x,
                y=content_center_y + 10,
                font_face=cv2.FONT_HERSHEY_DUPLEX,
                scale=0.9,
                color=self.COLOR_GRAY,
                thickness=2,
            )

    def draw_state_overlays(
        self, frame: np.ndarray, context: GameContext, is_pvp: bool
    ) -> None:
        """대형 결과 및 상태 애니메이션 렌더링"""
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2

        match context.state:
            case GameState.IDLE:
                help_text = (
                    "Press [SPACE] to Start | '1':PvP '2':PvE '3':GodMode 'R':Reset"
                )
                self._draw_centered_text(
                    frame,
                    help_text,
                    center_x,
                    h - 25,
                    scale=0.48,
                    color=self.COLOR_YELLOW,
                    thickness=1,
                )

            case GameState.COUNTDOWN:
                elapsed = time.time() - context.state_timer
                remain = max(0.0, self.countdown_sec - elapsed)
                count_str = str(int(np.ceil(remain))) if remain > 0.5 else "GO!"
                color = self.COLOR_YELLOW if remain > 0.5 else self.COLOR_GREEN

                self._draw_centered_text(
                    frame,
                    count_str,
                    center_x,
                    center_y + 25,
                    font_face=cv2.FONT_HERSHEY_DUPLEX,
                    scale=2.5,
                    color=color,
                    thickness=4,
                )

            case GameState.JUDGING:
                self._draw_centered_text(
                    frame,
                    "JUDGING...",
                    center_x,
                    center_y + 15,
                    font_face=cv2.FONT_HERSHEY_DUPLEX,
                    scale=1.2,
                    color=self.COLOR_BLUE,
                    thickness=3,
                )

            case GameState.ROUND_RESULT:
                # 1. 결과 문구 및 색상 매핑 (PvE 모드일 때 PC로 자동 변환)
                res_color = self.COLOR_GRAY
                if context.last_result == GameResult.PLAYER1_WIN:
                    res_text = "PLAYER 1 WIN!"
                    res_color = self.COLOR_GREEN
                elif context.last_result == GameResult.PLAYER2_WIN:
                    res_text = "PLAYER 2 WIN!" if is_pvp else "PC WIN!"
                    res_color = self.COLOR_RED
                elif context.last_result == GameResult.DRAW:
                    res_text = "DRAW"
                    res_color = self.COLOR_BLUE
                else:
                    res_text = "NO HAND DETECTED"
                    res_color = self.COLOR_GRAY

                # 2. 중앙 배너 오버레이 (적절한 상하 패딩)
                self._draw_overlay_rect(
                    frame,
                    0,
                    center_y - 75,
                    w,
                    center_y + 70,
                    self.COLOR_BG_DARK,
                    alpha=0.8,
                )

                # 3. 결과 문구 출력 (글자 크기 0.95로 최적화하여 넘침 방지)
                self._draw_centered_text(
                    frame,
                    res_text,
                    center_x,
                    center_y - 25,
                    font_face=cv2.FONT_HERSHEY_DUPLEX,
                    scale=0.95,
                    color=res_color,
                    thickness=2,
                )

                # 4. 양측 손 아이콘 및 VS 표시 (중심 좌표 기반 완전 대칭)
                self._overlay_icon(
                    frame,
                    context.p1_final_gesture,
                    center_x=center_x - 70,
                    center_y=center_y + 25,
                    size=60,
                )
                self._draw_centered_text(
                    frame,
                    "VS",
                    center_x=center_x,
                    y=center_y + 32,
                    font_face=cv2.FONT_HERSHEY_DUPLEX,
                    scale=0.65,
                    color=self.COLOR_WHITE,
                    thickness=2,
                )
                self._overlay_icon(
                    frame,
                    context.p2_final_gesture,
                    center_x=center_x + 70,
                    center_y=center_y + 25,
                    size=60,
                )

            case GameState.MATCH_OVER:
                winner = (
                    "PLAYER 1"
                    if context.p1_score > context.p2_score
                    else ("PLAYER 2" if is_pvp else "PC (AI)")
                )
                over_text = f"MATCH WINNER: {winner}!"
                self._draw_centered_text(
                    frame,
                    over_text,
                    center_x,
                    center_y - 10,
                    font_face=cv2.FONT_HERSHEY_DUPLEX,
                    scale=1.0,
                    color=self.COLOR_YELLOW,
                    thickness=2,
                )
                self._draw_centered_text(
                    frame,
                    "Press 'R' to Start a New Match",
                    center_x,
                    center_y + 35,
                    scale=0.55,
                    color=self.COLOR_WHITE,
                    thickness=1,
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
