import json
import time
from pathlib import Path
from collections import Counter


class AuditLogger:
    """
    Writes JSONL logs at ~1Hz and produces a summary JSON at the end.
    """

    def __init__(self, log_every_s: float = 1.0):
        self.path = Path("outputs/sessions")
        self.path.mkdir(parents=True, exist_ok=True)

        self.session_id = int(time.time())
        self.filepath = self.path / f"session_{self.session_id}.jsonl"
        self.file = open(self.filepath, "a", encoding="utf-8")

        self.log_every_s = log_every_s
        self._last_log_ts = 0.0

        self.start_ts = time.time()
        self.total_logs = 0
        self.decision_counts = Counter()
        self.reason_counts = Counter()
        self.face_detected_logs = 0
        self.brightness_vals = []
        self.blur_vals = []

    def maybe_log(self, data: dict) -> bool:
        now = time.time()
        if now - self._last_log_ts < self.log_every_s:
            return False

        self._last_log_ts = now
        self.file.write(json.dumps(data) + "\n")
        self.file.flush()

        self.total_logs += 1
        decision = data.get("decision")
        reason = data.get("reason")
        metrics = data.get("metrics") or {}

        if decision:
            self.decision_counts[decision] += 1
        if reason:
            self.reason_counts[reason] += 1

        if metrics:
            self.face_detected_logs += 1
            if "brightness" in metrics:
                self.brightness_vals.append(metrics["brightness"])
            if "blur" in metrics:
                self.blur_vals.append(metrics["blur"])

        return True

    def close(self):
        self.file.close()
        return self.write_summary()

    def write_summary(self):
        duration_s = time.time() - self.start_ts
        total = max(1, self.total_logs)

        def avg(xs):
            return (sum(xs) / len(xs)) if xs else None

        summary = {
            "session_id": self.session_id,
            "duration_s": round(duration_s, 2),
            "log_rate_hz": (1.0 / self.log_every_s) if self.log_every_s > 0 else None,
            "total_logs": self.total_logs,
            "decision_counts": dict(self.decision_counts),
            "reason_counts": dict(self.reason_counts),
            "pass_rate": self.decision_counts.get("PASS", 0) / total,
            "face_detected_rate": self.face_detected_logs / total,
            "avg_brightness": avg(self.brightness_vals),
            "avg_blur": avg(self.blur_vals),
            "top_reasons": self.reason_counts.most_common(5),
        }

        outpath = self.path / f"session_{self.session_id}_summary.json"
        outpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return summary, str(outpath), str(self.filepath)
