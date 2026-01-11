import time
from pathlib import Path

import streamlit as st
import cv2

from detector.face_detector import FaceDetector
from quality.metrics import compute_metrics
from quality.rules import evaluate
from utils.config import THRESHOLDS
from LLM.llm import get_llm_advice, get_template_advice
from audit.audit_logger import AuditLogger


def draw_overlay(frame, det, decision, reason, fps):
    h, w = frame.shape[:2]

    # Draw face bounding box
    if det and "bbox" in det:
        x, y, bw, bh = det["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    # Decision text
    cv2.putText(
        frame,
        f"Decision: {decision}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # Reason text
    if reason:
        cv2.putText(
            frame,
            f"Reason: {reason}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Center crosshair
    cv2.drawMarker(
        frame,
        (w // 2, h // 2),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=1,
    )

    return frame


# ---------------- STREAMLIT UI ----------------

st.set_page_config(layout="wide")
st.title("AI Camera Quality Gate (KYC)")

# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "last_reason" not in st.session_state:
    st.session_state.last_reason = None
if "last_advice_ts" not in st.session_state:
    st.session_state.last_advice_ts = 0.0
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "logger" not in st.session_state:
    st.session_state.logger = None

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Controls")

    start = st.button("Start Camera", disabled=st.session_state.running)
    stop = st.button("Stop Camera", disabled=not st.session_state.running)

    use_llm = st.checkbox("Use LLM for advice (optional)", value=False)
    cam_index = st.number_input(
        "Camera index", min_value=0, max_value=5, value=0, step=1
    )

    st.caption("Capture is enabled only when decision == PASS.")

with col2:
    frame_window = st.image([])
    status = st.empty()
    advice_box = st.empty()

    cap_col1, cap_col2 = st.columns([1, 1])
    capture_btn = cap_col1.button("Capture (PASS only)")
    capture_note = cap_col2.empty()

# Start / Stop logic
if start:
    st.session_state.running = True
    st.session_state.logger = AuditLogger(log_every_s=1.0)

if stop:
    st.session_state.running = False

detector = FaceDetector()

# ---------------- MAIN LOOP ----------------

if st.session_state.running:
    cap = cv2.VideoCapture(int(cam_index))

    if not cap.isOpened():
        st.error("Could not open camera.")
        st.session_state.running = False
    else:
        prev_time = time.time()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame.")
                break

            now = time.time()
            dt = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

            det = detector.detect(frame)

            if not det:
                decision, reason = "FAIL", "no_face"
                metrics = {}
            else:
                metrics = compute_metrics(frame, det["bbox"], det["confidence"])
                det["bbox"] = metrics.get("bbox", det["bbox"])
                decision, reason = evaluate(metrics, THRESHOLDS)

            # Advice anti-spam
            if reason:
                if (
                    reason != st.session_state.last_reason
                    or now - st.session_state.last_advice_ts > 3
                ):
                    advice = (
                        get_llm_advice(reason, metrics)
                        if use_llm
                        else get_template_advice(reason)
                    )
                    st.session_state.last_reason = reason
                    st.session_state.last_advice_ts = now
                else:
                    advice = get_template_advice(reason)
            else:
                advice = "Looks good. You can capture now."

            overlay = draw_overlay(frame.copy(), det, decision, reason, fps)
            frame_window.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

            st.session_state.last_frame = frame.copy()

            status.metric("Decision", decision)
            advice_box.info(advice)

            # Capture gating
            if capture_btn:
                if decision == "PASS" and st.session_state.last_frame is not None:
                    outdir = Path("outputs/captures")
                    outdir.mkdir(parents=True, exist_ok=True)
                    fname = outdir / f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(str(fname), st.session_state.last_frame)
                    capture_note.success(f"Saved: {fname}")
                else:
                    capture_note.error("Capture disabled: need PASS.")

            # Audit logging (1Hz)
            if st.session_state.logger:
                st.session_state.logger.maybe_log(
                    {
                        "ts": now,
                        "decision": decision,
                        "reason": reason,
                        "metrics": metrics,
                        "fps": round(fps, 2),
                    }
                )

            time.sleep(0.01)

        cap.release()

        if st.session_state.logger:
            summary, summary_path, jsonl_path = st.session_state.logger.close()
            st.session_state.logger = None
            st.session_state.running = False

            st.success("Session ended.")
            st.write("Session files:")
            st.write(f"- JSONL: {jsonl_path}")
            st.write(f"- Summary: {summary_path}")
            st.subheader("Session Summary")
            st.json(summary)
