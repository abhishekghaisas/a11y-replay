from a11y_replay.events import ChangeEvent, KeyEvent
from a11y_replay.sync import estimate_offset, estimate_offset_search, to_log_clock

KEYS = [KeyEvent(t=t, key="Tab") for t in (1.0, 2.5, 4.0, 5.2, 6.5)]


def _changes(shift, extra=()):
    ts = [k.t + shift for k in KEYS] + list(extra)
    return [ChangeEvent(t=t, t_end=t, bbox=(0, 0, 10, 10), magnitude=0.01, n_frames=1) for t in ts]


def test_recovers_known_offsets():
    for true in (-0.3, -0.07, 0.0, 0.3):
        est = estimate_offset(KEYS, _changes(true))
        assert est.ok and abs(est.offset - true) < 1e-6, (true, est)


def test_ignores_unrelated_far_changes():
    est = estimate_offset(KEYS, _changes(-0.07, extra=(10.0, 12.5)))
    assert est.ok and abs(est.offset + 0.07) < 1e-6 and est.n_pairs == 5


def test_median_is_robust_to_one_bad_pair():
    changes = _changes(-0.07)
    changes[2] = ChangeEvent(t=KEYS[2].t + 0.5, t_end=KEYS[2].t + 0.5,
                             bbox=(0, 0, 1, 1), magnitude=0.01, n_frames=1)
    assert abs(estimate_offset(KEYS, changes).offset + 0.07) < 1e-6


def test_no_effects_means_not_ok_and_zero_offset():
    est = estimate_offset(KEYS, [])
    assert est.n_pairs == 0 and not est.ok and est.offset == 0.0


def test_too_few_pairs_not_ok():
    est = estimate_offset(KEYS, _changes(-0.07)[:2])
    assert est.n_pairs == 2 and not est.ok


def test_inconsistent_pairs_not_ok():
    ts = [k.t + d for k, d in zip(KEYS, (-0.3, 0.5, -0.2, 0.4, 0.0))]
    changes = [ChangeEvent(t=t, t_end=t, bbox=(0, 0, 1, 1), magnitude=0.01, n_frames=1) for t in ts]
    assert not estimate_offset(KEYS, changes).ok


def test_to_log_clock_maps_with_offset():
    c = ChangeEvent(t=1.0, t_end=1.5, bbox=(0, 0, 1, 1), magnitude=0.1, n_frames=2)
    out = to_log_clock([c], offset=-0.07)[0]
    assert abs(out.t - 1.07) < 1e-9 and abs(out.t_end - 1.57) < 1e-9


IRREGULAR = [KeyEvent(t=t, key="F9") for t in (10.0, 11.1, 12.6, 13.4, 16.0, 18.3)]


def _ch(t):
    return ChangeEvent(t=t, t_end=t, bbox=(0, 0, 10, 10), magnitude=0.01, n_frames=1)


def test_local_estimator_cannot_see_a_recording_that_started_seconds_apart():
    changes = [_ch(k.t + 7.3) for k in IRREGULAR]
    assert not estimate_offset(IRREGULAR, changes).ok            # the local window is under a second


def test_search_recovers_large_positive_and_negative_offsets():
    for true in (7.3, -12.5, 0.05, 45.0):
        changes = [_ch(k.t + true + 0.02) for k in IRREGULAR]
        est = estimate_offset_search(IRREGULAR, changes)
        assert est.ok and abs(est.offset - (true + 0.02)) < 0.06, (true, est)


def test_search_ignores_unrelated_changes():
    rng = [3.7, 21.9, 25.2, 31.4, 33.3, 40.8, 52.1, 58.6, 61.0]            # decoys: page loads, scrolling...
    changes = [_ch(k.t + 7.3) for k in IRREGULAR] + [_ch(t) for t in rng]
    est = estimate_offset_search(IRREGULAR, changes)
    assert est.ok and abs(est.offset - 7.3) < 0.06 and est.n_pairs >= 5


def test_search_with_nothing_to_pair_is_not_ok():
    assert not estimate_offset_search(IRREGULAR, []).ok
    assert not estimate_offset_search([], [_ch(3.0)]).ok


def test_every_pair_is_recorded_with_its_delta_in_real_clock_terms():
    keys = IRREGULAR[:5]
    est = estimate_offset_search(keys, [_ch(k.t + 7.3 + j) for k, j in zip(keys, (0.0, 0.01, -0.01, 0.0, 0.02))])
    assert len(est.detail) == 5
    for (name, kt, delta, frac), k in zip(est.detail, keys):
        assert name == "F9" and abs(kt - k.t) < 1e-6 and abs(delta - 7.3) < 0.05   # key time is the logged one, not shifted


def test_each_pair_records_how_big_the_matched_change_was():
    keys = [KeyEvent(t=1.0, key="F9"), KeyEvent(t=2.5, key="F9"), KeyEvent(t=4.0, key="F9")]
    big = ChangeEvent(t=1.05, t_end=1.2, bbox=(0, 0, 1000, 700), magnitude=0.28, n_frames=3)
    small = [ChangeEvent(t=2.52, t_end=2.7, bbox=(0, 0, 120, 120), magnitude=0.015, n_frames=3),
             ChangeEvent(t=4.02, t_end=4.2, bbox=(0, 0, 120, 120), magnitude=0.015, n_frames=3)]
    est = estimate_offset(keys, [big] + small)
    assert [round(d[3], 3) for d in est.detail] == [0.28, 0.015, 0.015]


def test_the_estimate_records_which_keys_were_matched():
    keys = [KeyEvent(t=1.0, key="F9"), KeyEvent(t=2.5, key="F9"), KeyEvent(t=4.0, key="Tab"), KeyEvent(t=5.2, key="Tab")]
    est = estimate_offset(keys, _changes(0.0)[:4])
    assert est.keys_used == ("F9", "F9", "Tab", "Tab")
    assert estimate_offset_search(keys, _changes(7.0)[:4]).keys_used == ("F9", "F9", "Tab", "Tab")