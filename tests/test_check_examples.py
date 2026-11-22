import hashlib
import shutil

from a11y_replay.check_examples import fix, main, parse_checksums, scan


def _make(tmp_path):
    files = {"demo_examples/a/video.mp4": b"AAAA", "demo_examples/a/keys.jsonl": b"ka",
             "demo_examples/b/video.mp4": b"BBBB", "demo_examples/b/keys.jsonl": b"kb",
             "demo_examples/c/utterances.json": b"[]"}
    lines = []
    for rel, data in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    (tmp_path / "demo_examples" / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    (tmp_path / "demo_examples" / "README.md").write_text("notes")
    return files


def test_a_correct_set_is_all_ok(tmp_path):
    _make(tmp_path)
    r = scan(tmp_path, parse_checksums(tmp_path / "demo_examples" / "CHECKSUMS.sha256"))
    assert len(r["ok"]) == 5 and not r["misplaced"] and not r["unknown"] and not r["missing"]
    assert main(["--root", str(tmp_path)]) == 0


def test_swapped_files_are_recognized_by_content_and_fixed_without_loss(tmp_path):
    files = _make(tmp_path)
    a, b = tmp_path / "demo_examples/a/video.mp4", tmp_path / "demo_examples/b/video.mp4"
    tmp = tmp_path / "x"
    shutil.move(a, tmp); shutil.move(b, a); shutil.move(tmp, b)                     # the real-world mistake: two same-named files swapped
    cs = parse_checksums(tmp_path / "demo_examples" / "CHECKSUMS.sha256")
    r = scan(tmp_path, cs)
    assert sorted(r["misplaced"]) == [("demo_examples/a/video.mp4", "demo_examples/b/video.mp4"),
                                      ("demo_examples/b/video.mp4", "demo_examples/a/video.mp4")]
    assert main(["--root", str(tmp_path)]) == 1                                      # report only: nothing moved yet
    assert a.read_bytes() == b"BBBB"
    assert main(["--root", str(tmp_path), "--fix"]) == 0
    assert all((tmp_path / rel).read_bytes() == data for rel, data in files.items())


def test_a_download_saved_under_a_browser_renamed_name_is_moved_into_place(tmp_path):
    _make(tmp_path)
    shutil.move(tmp_path / "demo_examples/a/keys.jsonl", tmp_path / "demo_examples/a/keys (1).jsonl")
    r = scan(tmp_path, parse_checksums(tmp_path / "demo_examples" / "CHECKSUMS.sha256"))
    assert r["misplaced"] == [("demo_examples/a/keys (1).jsonl", "demo_examples/a/keys.jsonl")] and r["missing"] == []
    assert main(["--root", str(tmp_path), "--fix"]) == 0
    assert (tmp_path / "demo_examples/a/keys.jsonl").read_bytes() == b"ka"
    assert not (tmp_path / "demo_examples/a/keys (1).jsonl").exists()


def test_files_that_match_nothing_are_reported_and_never_overwritten_or_deleted(tmp_path):
    _make(tmp_path)
    (tmp_path / "demo_examples/a/video.mp4").write_bytes(b"MY OWN RECORDING")        # regenerated or modified
    r = scan(tmp_path, parse_checksums(tmp_path / "demo_examples" / "CHECKSUMS.sha256"))
    assert r["unknown"] == ["demo_examples/a/video.mp4"] and r["missing"] == ["demo_examples/a/video.mp4"]
    assert main(["--root", str(tmp_path), "--fix"]) == 1
    assert (tmp_path / "demo_examples/a/video.mp4").read_bytes() == b"MY OWN RECORDING"


def test_fix_refuses_to_overwrite_an_unaccounted_file(tmp_path):
    _make(tmp_path)
    stray = tmp_path / "demo_examples/b/video (2).mp4"
    shutil.move(tmp_path / "demo_examples/b/video.mp4", stray)                        # b's video is saved under a stray name ...
    (tmp_path / "demo_examples/b/video.mp4").write_bytes(b"SOMETHING ELSE")           # ... and something unknown sits where it belongs
    r = scan(tmp_path, parse_checksums(tmp_path / "demo_examples" / "CHECKSUMS.sha256"))
    moves, refused = fix(tmp_path, r)
    assert moves == [] and refused and (tmp_path / "demo_examples/b/video.mp4").read_bytes() == b"SOMETHING ELSE"
    assert stray.exists()