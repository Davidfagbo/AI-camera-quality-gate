from quality import reasons


def evaluate(metrics, t):
    """
    Deterministic rules:
    returns (decision, reason)
    """
    if metrics["face_confidence"] < t["min_conf"]:
        return "FAIL", reasons.LOW_CONFIDENCE

    if metrics["face_area_ratio"] < t["min_area"]:
        return "NEEDS_FIXING", reasons.TOO_FAR
    if metrics["face_area_ratio"] > t["max_area"]:
        return "NEEDS_FIXING", reasons.TOO_CLOSE

    if metrics["dx"] > t["max_dx"] or metrics["dy"] > t["max_dy"]:
        return "NEEDS_FIXING", reasons.OFF_CENTER

    if metrics["brightness"] < t["min_light"]:
        return "NEEDS_FIXING", reasons.LOW_LIGHT
    if metrics["brightness"] > t["max_light"]:
        return "NEEDS_FIXING", reasons.TOO_BRIGHT

    if metrics["blur"] < t["min_blur"]:
        return "NEEDS_FIXING", reasons.BLURRY

    return "PASS", None
