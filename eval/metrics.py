"""Event-level precision / recall / F1 per issue type with a time tolerance."""
from a11y_replay.config import DEFAULT


def match_events(pred_times: list[float], true_times: list[float],
                 tol: float = DEFAULT.time_tolerance_s) -> tuple[int, int, int]:
    """Greedy one-to-one matching. Returns (tp, fp, fn)."""
    unmatched = sorted(true_times)
    tp = 0
    for p in sorted(pred_times):
        hit = next((t for t in unmatched if abs(t - p) <= tol), None)
        if hit is not None:
            unmatched.remove(hit)
            tp += 1
    return tp, len(pred_times) - tp, len(unmatched)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
