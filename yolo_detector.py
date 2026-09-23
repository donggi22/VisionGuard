import numpy as np
import cv2
from ultralytics import YOLO

from config import YOLO_MODEL, YOLO_CONFIDENCE, ALERT_CLASSES


class Detection:
    __slots__ = ("label", "confidence", "box")

    def __init__(self, label: str, confidence: float, box: tuple):
        self.label = label
        self.confidence = confidence
        self.box = box  # (x1, y1, x2, y2)


class YoloDetector:
    def __init__(self):
        self._model = YOLO(YOLO_MODEL)
        self._alert_classes = set(ALERT_CLASSES)

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """여러 프레임을 한 번에 추론. 프레임별 알림 대상 클래스 감지 결과를 반환."""
        out: list[list[Detection]] = []
        for results in self._model(frames, conf=YOLO_CONFIDENCE, verbose=False):
            detections: list[Detection] = []
            for box in results.boxes:
                label = self._model.names[int(box.cls)]
                if label not in self._alert_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(label, float(box.conf), (x1, y1, x2, y2)))
            out.append(detections)
        return out

    @staticmethod
    def draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        color = (0, 0, 255)
        for d in detections:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{d.label} {d.confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame
