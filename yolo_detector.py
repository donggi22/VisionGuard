import numpy as np
import cv2
from ultralytics import YOLO

from config import YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IMGSZ, ALERT_CLASSES


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

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """이번 프레임에서 ALERT_CLASSES에 해당하는 객체를 감지."""
        results = self._model(frame, conf=YOLO_CONFIDENCE, imgsz=YOLO_IMGSZ, verbose=False)[0]
        detections: list[Detection] = []

        for box in results.boxes:
            label = self._model.names[int(box.cls)]
            if label not in self._alert_classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(Detection(label, float(box.conf), (x1, y1, x2, y2)))

        return detections

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
