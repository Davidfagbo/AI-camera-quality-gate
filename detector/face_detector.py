import cv2
import mediapipe as mp


class FaceDetector:
    """
    Face detector using MediaPipe Face Detection.
    Returns best face bbox + confidence per frame.
    """

    def __init__(self, min_conf: float = 0.6):
        self.mp_fd = mp.solutions.face_detection
        self.detector = self.mp_fd.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_conf,
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)

        if not result.detections:
            return None

        best = max(result.detections, key=lambda d: d.score[0])
        conf = float(best.score[0])

        box = best.location_data.relative_bounding_box
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        return {"bbox": (x, y, bw, bh), "confidence": conf}
