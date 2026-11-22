from eval.metrics import match_events, prf


def test_match_events_basic():
    tp, fp, fn = match_events([1.0, 5.0], [1.4, 9.0], tol=1.0)
    assert (tp, fp, fn) == (1, 1, 1)


def test_prf_zero_safe():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
