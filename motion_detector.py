import cv2
import numpy as np
from config import MOTION_THRESHOLD, MOTION_MIN_FRAMES


class MotionDetector:
    def __init__(self, roi: tuple = None):
        """
        roi: (x, y, w, h) — 감지 구역 지정. None이면 전체 화면.
        """
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=35, detectShadows=True
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

        # 그림자는 127로 표시되므로 임계값을 높여 순수 객체(255)만 남김
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 자잘한 빛 노이즈 다수 합산 방지: 500px² 이상인 의미 있는 덩어리만 집계
        meaningful_contours = [c for c in contours if cv2.contourArea(c) > 500]
        motion_area = sum(cv2.contourArea(c) for c in meaningful_contours)

        if motion_area >= MOTION_THRESHOLD:
            if self._consecutive == 0:
                print(f"[모션감지-진단] 임계값 통과 area={motion_area:.0f} (threshold={MOTION_THRESHOLD})")
            self._consecutive += 1
        else:
            if self._consecutive > 0:
                print(
                    f"[모션감지-진단] {self._consecutive}프레임 만에 소실"
                    f" (min_frames={MOTION_MIN_FRAMES} 못 채움, area={motion_area:.0f})"
                )
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
