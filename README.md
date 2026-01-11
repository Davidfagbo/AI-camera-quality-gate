# AI Camera Quality Gate

A real-time webcam application that evaluates **camera and image quality** for identity verification.

The system detects a face, computes objective quality metrics, applies **deterministic rules** to decide whether an image is usable, and provides **one-line corrective guidance** (template-based or optional LLM-assisted). All decisions are logged for auditability.

This project does **not** perform face recognition or identity inference.

---

## Problem Statement

Low-quality selfie images (poor lighting, blur, incorrect framing) are a major cause of identity verification failures in:
- banking onboarding
- remote exams
- online identity checks

This project implements a **camera quality gate** that blocks capture until quality requirements are met.

---

## System Pipeline

Camera frame  
→ Face detection  
→ Quality metrics  
→ Deterministic rules  
→ PASS / NEEDS_FIXING / FAIL  
→ User guidance  
→ Audit logs

The LLM is used **only to explain decisions**, never to make them.

---

## Project Structure

AI_cam/
- app.py (Streamlit entry point)
- detector/ (face detection)
- quality/ (metrics and rules)
- utils/ (threshold configuration)
- LLM/ (template + optional LLM guidance)
- audit/ (JSONL logs and summaries)
- outputs/ (generated files, gitignored)
- requirements.txt
- README.md

---

## Quality Metrics

Computed per frame on the detected face region:
- Face area ratio (distance proxy)
- Centering offset (dx, dy)
- Brightness (mean grayscale value)
- Blur (variance of Laplacian)
- Detection confidence

---

## Decision Rules

Rules are evaluated top-down. Only one primary reason is shown.

- No face detected → FAIL
- Face too far / too close → NEEDS_FIXING
- Off-center → NEEDS_FIXING
- Poor lighting → NEEDS_FIXING
- Blurry image → NEEDS_FIXING
- All checks pass → PASS

---

## Capture Gating

Image capture is enabled **only when decision == PASS**, mimicking real KYC flows.

Captured images and logs are written to the outputs/ directory (ignored by Git).

---

## Audit Logging

Each session logs:
- timestamp
- decision
- reason
- metrics
- FPS

A session summary is produced at the end.

---

## Safety and Ethics

- No identity recognition
- No demographic inference
- Local-only processing
- Deterministic, explainable decisions

---

## Running Locally

1. Create environment  
python3.11 -m venv .venv  
source .venv/bin/activate  

2. Install dependencies  
pip install -r requirements.txt  

3. (Optional) Enable LLM  
Create .env with OPENAI_API_KEY

4. Run  
streamlit run app.py

---

## License

Educational / demonstration project.