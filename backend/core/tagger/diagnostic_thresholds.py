"""Select diagnostic decision thresholds without recalibrating captured logits."""
import numpy as np


def decision_thresholds(snapshot, settings):
    size = len(snapshot["probabilities"])
    mode = settings["threshold_mode"]
    fallback = float(snapshot["threshold"] if mode == "training" else settings["threshold"])
    result = np.full(size, fallback, dtype=np.float32)
    eligible = np.zeros(size, dtype=bool)
    if mode == "per_tag" and snapshot.get("tag_metrics") is not None:
        metrics = snapshot["tag_metrics"]
        best, support = (np.asarray(metrics[key]) for key in ("best_thr", "n_pos"))
        if best.shape != (size,) or support.shape != (size,):
            raise ValueError("Captured tag threshold vocabulary differs")
        eligible = np.isfinite(best) & (best >= 0) & (best <= 1) & (support >= settings["min_samples_for_per_tag"])
        result[eligible] = np.maximum(best[eligible], settings["min_best_thr"])
    if not 0 <= fallback <= 1:
        raise ValueError("Invalid captured common threshold")
    return result, fallback, int(eligible.sum())
