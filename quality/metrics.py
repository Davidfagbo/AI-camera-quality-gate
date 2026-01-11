import cv2
import numpy as np


def _clamp_bbox(x, y, bw, bh, w, h):
    x = max(0, int(x))
    y = max(0, int(y))
    bw = max(1, int(bw))
    bh = max(1, int(bh))

    if x + bw > w:
        bw = max(1, w - x)
    if y + bh > h:
        bh = max(1, h - y)

    return x, y, bw, bh


def compute_metrics(frame, bbox, confidence):
    """
    Compute quality metrics on face ROI.
    Returns dict with metrics + clamped bbox (useful for drawing).
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    x, y, bw, bh = _clamp_bbox(x, y, bw, bh, w, h)

    roi = frame[y : y + bh, x : x + bw]
    if roi.size == 0:
        return {
            "face_confidence": float(confidence),
            "face_area_ratio": 0.0,
            "dx": 1.0,
            "dy": 1.0,
            "brightness": 0.0,
            "blur": 0.0,
            "bbox": (x, y, bw, bh),
        }

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    face_area_ratio = (bw * bh) / (w * h)

    cx = x + bw / 2
    cy = y + bh / 2
    dx = abs(cx - w / 2) / w
    dy = abs(cy - h / 2) / h

    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "face_confidence": float(confidence),
        "face_area_ratio": float(face_area_ratio),
        "dx": float(dx),
        "dy": float(dy),
        "brightness": brightness,
        "blur": blur,
        "bbox": (x, y, bw, bh),
    }
