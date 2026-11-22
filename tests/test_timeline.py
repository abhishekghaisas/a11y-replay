from a11y_replay.events import ChangeEvent, KeyEvent, Utterance
from a11y_replay.timeline import Timeline


def _tl():
    keys = [KeyEvent(t=1.0, key="Tab"), KeyEvent(t=3.0, key="Tab")]
    changes = [ChangeEvent(t=0.93, t_end=0.93, bbox=(0, 0, 5, 5), magnitude=0.01, n_frames=1),
               ChangeEvent(t=5.0, t_end=5.2, bbox=(0, 0, 5, 5), magnitude=0.02, n_frames=3)]
    speech = [Utterance(start=1.4, end=2.0, text="Name")]
    return Timeline.build(keys, changes, speech, video_offset=-0.07)


def test_items_sorted_and_counted():
    tl = _tl()
    assert [i.kind for i in tl] == ["key", "change", "speech", "key", "change"]
    ts = [i.t for i in tl]
    assert ts == sorted(ts)
    assert len(tl.items("key")) == 2 and len(tl.items("change")) == 2 and len(tl.items("speech")) == 1


def test_video_offset_applied_to_changes_only():
    tl = _tl()
    first_change = tl.items("change")[0]
    assert abs(first_change.t - 1.00) < 1e-9  # 0.93 - (-0.07)
    assert tl.items("key")[0].t == 1.0 and tl.items("speech")[0].t == 1.4


def test_effects_of_key_finds_its_change_not_others():
    tl = _tl()
    k1, k2 = tl.items("key")
    assert len(tl.effects_of(k1)) == 1
    assert tl.effects_of(k2) == []


def test_window_and_after_exclude_self():
    tl = _tl()
    k1 = tl.items("key")[0]
    assert all(i is not k1 for i in tl.after(k1, within=5.0))
    assert [i.kind for i in tl.window(1.3, 2.5)] == ["speech"]