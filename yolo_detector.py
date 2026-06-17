import time
import numpy as np
import cv2
from ultralytics import YOLO

from config import YOLO_MODEL, YOLO_CONFIDENCE, ALERT_CLASSES, ALERT_COOLDOWN_SECONDS


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
        # {label: last_alert_timestamp}
        self._last_alert: dict[str, float] = {}
        # 이전 프레임에서 감지된 클래스 집합
        self._prev_labels: set[str] = set()

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], list[Detection]]:
        """
        Returns:
            detections: 이번 프레임 전체 감지 결과
            new_objects: 이전 프레임에 없던 & 쿨타임 지난 객체
        """
        results = self._model(frame, conf=YOLO_CONFIDENCE, verbose=False)[0]
        detections: list[Detection] = []

        for box in results.boxes:
            label = self._model.names[int(box.cls)]
            if label not in self._alert_classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(Detection(label, float(box.conf), (x1, y1, x2, y2)))

        current_labels = {d.label for d in detections}
        now = time.time()

        new_objects: list[Detection] = []
        for d in detections:
            cooldown_ok = (now - self._last_alert.get(d.label, 0)) >= ALERT_COOLDOWN_SECONDS
            if cooldown_ok:
                new_objects.append(d)
                self._last_alert[d.label] = now

        self._prev_labels = current_labels
        return detections, new_objects

    @staticmethod
    def draw(frame: np.ndarray, detections: list[Detection], highlight: set[str] = None) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d.box
            is_new = highlight and d.label in highlight
            color = (0, 0, 255) if is_new else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{d.label} {d.confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame
