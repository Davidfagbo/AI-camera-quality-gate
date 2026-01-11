import os
from openai import OpenAI


TEMPLATES = {
    "no_face": "Show your face clearly in the frame.",
    "low_confidence": "Improve lighting and face the camera.",
    "too_far": "Move closer to the camera.",
    "too_close": "Move a little farther from the camera.",
    "off_center": "Center your face in the frame.",
    "low_light": "Increase lighting or face a window.",
    "too_bright": "Reduce glare; avoid strong backlight.",
    "blurry": "Hold still and clean your camera lens.",
}

SYSTEM_PROMPT = (
    "You are a camera setup assistant for identity verification. "
    "Give one short actionable instruction. "
    "Do not infer identity or demographics. "
    "Max 12 words."
)


def get_template_advice(reason: str) -> str:
    if not reason:
        return ""
    return TEMPLATES.get(reason, "Adjust your camera setup and try again.")


def get_llm_advice(reason, metrics):
    fallback = get_template_advice(reason)
    if not reason:
        return ""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback

    client = OpenAI(api_key=api_key)

    user_msg = {
        "issue": reason,
        "metrics": {
            "brightness": round(metrics.get("brightness", 0.0), 1),
            "blur": round(metrics.get("blur", 0.0), 1),
            "dx": round(metrics.get("dx", 0.0), 3),
            "dy": round(metrics.get("dy", 0.0), 3),
            "face_area_ratio": round(metrics.get("face_area_ratio", 0.0), 3),
        },
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-5o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(user_msg)},
            ],
        )
        text = resp.choices[0].message.content.strip()
        return text if text else fallback
    except Exception:
        return fallback
