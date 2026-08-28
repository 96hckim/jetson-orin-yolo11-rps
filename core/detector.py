# 고속 전/후처리 및 YOLO 검출 파이프라인
import cv2
import numpy as np
from configs.app_config import VisionConfig
from game.types import Detection, Gesture

from core.trt_engine import TRTEngine


class YOLOv11Detector:
    """YOLOv11 TensorRT 객체 탐지 파이프라인"""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.engine = TRTEngine(config.engine_path)
        self.input_size = config.input_size

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        target_shape: tuple[int, int] = (320, 320),
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, int, int]:
        """종횡비를 유지하며 패딩을 추가하는 고속 리사이즈 함수"""
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = target_shape
        ratio = min(target_w / orig_w, target_h / orig_h)

        new_w, new_h = round(orig_w * ratio), round(orig_h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_x, pad_y = pad_w // 2, pad_h // 2
        right = pad_w - pad_x
        bottom = pad_h - pad_y

        padded = cv2.copyMakeBorder(
            resized,
            pad_y,
            bottom,
            pad_x,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )
        return padded, ratio, pad_x, pad_y

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """BGR 프레임을 1x3xHxW 정규화 Tensor로 변환"""
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded_img, ratio, pad_x, pad_y = self._letterbox(rgb_img, self.input_size)

        # HWC -> CHW 변환 및 메모리 연속성 확보
        tensor = padded_img.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        tensor = np.ascontiguousarray(tensor[None, ...])
        return tensor, ratio, pad_x, pad_y

    def postprocess(
        self,
        raw_output: np.ndarray,
        orig_shape: tuple[int, int],
        ratio: float,
        pad_x: int,
        pad_y: int,
    ) -> list[Detection]:
        """NumPy 벡터화 및 OpenCV C++ NMS 기반 초고속 후처리"""
        orig_h, orig_w = orig_shape

        # (1, 7, 2100) -> (2100, 7) 형상 복원
        # 구조: [cx, cy, w, h, Paper, Rock, Scissors]
        detections_raw = raw_output.reshape(self.engine.output_shape)[0].T

        cls_scores = detections_raw[:, 4:7]
        scores = np.max(cls_scores, axis=1)
        class_ids = np.argmax(cls_scores, axis=1)

        # 1차 Confidence 임계값 필터링
        mask = scores >= self.config.conf_threshold
        if not np.any(mask):
            return []

        boxes = detections_raw[mask, :4]
        filtered_scores = scores[mask]
        filtered_classes = class_ids[mask]

        # Letterbox 역연산 -> [x1, y1, width, height] 변환
        cx, cy, w, h = (
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2],
            boxes[:, 3],
        )
        x1 = np.clip((cx - w / 2.0 - pad_x) / ratio, 0, orig_w)
        y1 = np.clip((cy - h / 2.0 - pad_y) / ratio, 0, orig_h)
        box_w = np.clip(w / ratio, 0, orig_w)
        box_h = np.clip(h / ratio, 0, orig_h)

        boxes_xywh = np.stack([x1, y1, box_w, box_h], axis=1).astype(int).tolist()
        scores_list = filtered_scores.tolist()

        # OpenCV C++ NMS 실행
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores_list,
            self.config.conf_threshold,
            self.config.iou_threshold,
        )

        if len(indices) == 0:
            return []

        results: list[Detection] = []
        for idx in np.array(indices).flatten():
            bx, by, bw, bh = boxes_xywh[idx]
            results.append(
                Detection(
                    gesture=Gesture(int(filtered_classes[idx])),
                    confidence=float(filtered_scores[idx]),
                    bbox=(bx, by, bx + bw, by + bh),
                )
            )

        return results

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """단일 프레임 엔드투엔드 추론 인터페이스"""
        tensor, ratio, pad_x, pad_y = self.preprocess(frame)
        raw_output = self.engine.infer(tensor)
        return self.postprocess(raw_output, frame.shape[:2], ratio, pad_x, pad_y)
