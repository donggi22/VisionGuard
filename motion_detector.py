import cv2
import numpy as np
from config import MOTION_THRESHOLD, MOTION_MIN_FRAMES


class MotionDetector:
    def __init__(self, roi: tuple = None):
        """
        roi: (x, y, w, h) — 감지 구역 지정. None이면 전체 화면.
        """
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )
        self._roi = roi
        self._consecutive = 0  # 연속 모션 프레임 카운터
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Returns:
            motion_detected (bool)
            mask (np.ndarray): 시각화용 마스크
        """
        work = frame if self._roi is None else self._crop(frame)

        mask = self._bg_subtractor.apply(work)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours)

        if motion_area >= MOTION_THRESHOLD:
            self._consecutive += 1
        else:
            self._consecutive = 0

        detected = self._consecutive >= MOTION_MIN_FRAMES
        return detected, mask

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self._roi
        return frame[y:y+h, x:x+w]

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        if self._roi is None:
            return frame
        x, y, w, h = self._roi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, "ROI", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame
