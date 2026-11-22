import shutil
import subprocess
from pathlib import Path

from a11y_replay.prepush import check, classify, main

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "sk-ant-" + "a1B2" * 8                    # built at runtime so this file does not contain a key-shaped string


def _repo(tmp_path, with_gitignore=True):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    if with_gitignore:
        shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    (tmp_path / "README.md").write_text("hello")
    return tmp_path


def _put(root, rel, data=b"x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())


def _blocked(r):
    return {rel for rel, _ in r["block"]}


def test_path_rules():
    assert classify("recordings/a/video.mov") and classify("benchmark/data/x/video.mp4") and classify(".env")
    assert classify("a/b/audit_report/report.html") and classify(".DS_Store") and classify("x/.streamlit/secrets.toml".split("x/")[1])
    assert classify("video.mov") and classify("clip.wav") and classify("foo/clip.mp4")
    assert classify("demo_examples/focus_page_clean/video.mp4") is None and classify("docs/examples/a.png") is None
    assert classify("a11y_replay/audit.py") is None and classify(".env.example") is None
    assert classify("benchmark/data/.gitkeep") is None and classify("eval/results/.gitkeep") is None   # placeholders are fine


def test_with_the_projects_gitignore_private_things_never_become_candidates(tmp_path):
    root = _repo(tmp_path)
    for rel in ("recordings/r/video.mov", "recordings/r/audit_report/report.html", ".env", "video.mov", ".DS_Store",
                ".streamlit/secrets.toml", "demo_examples/a/audit_report/report.html", "benchmark/data/x/video.mp4"):
        _put(root, rel)
    _put(root, ".streamlit/config.toml", "[server]\n")
    r = check(root)
    assert r["block"] == []
    assert ".streamlit/config.toml" in r["files"] and "recordings/r/video.mov" not in r["files"] and ".env" not in r["files"]


def test_without_a_gitignore_the_same_things_are_blocked_by_name(tmp_path):
    root = _repo(tmp_path, with_gitignore=False)
    for rel in ("recordings/r/video.mov", ".env", ".DS_Store", "x/audit_report/report.html", "clip.wav"):
        _put(root, rel)
    assert _blocked(check(root)) == {"recordings/r/video.mov", ".env", ".DS_Store", "x/audit_report/report.html", "clip.wav"}
    assert main([str(root)]) == 1


def test_a_key_inside_a_file_is_blocked_but_a_placeholder_is_not(tmp_path):
    root = _repo(tmp_path)
    _put(root, "notes.md", f"my key is {FAKE_KEY}")
    _put(root, "docs/setup.md", "Put ANTHROPIC_API_KEY=your-key-here in .env\nand ANTHROPIC_API_KEY=sk-ant-... also")
    _put(root, "cfg.py", f'ANTHROPIC_API_KEY = "{FAKE_KEY}"')
    b = check(root)["block"]
    assert {r for r, _ in b} == {"notes.md", "cfg.py"} and all("Anthropic API key" in w or "ANTHROPIC_API_KEY" in w for _, w in b)


def test_home_paths_and_big_files_warn_but_do_not_block(tmp_path):
    root = _repo(tmp_path)
    _put(root, "docs/log.md", "ran in /Us" + "ers/jane/projects/a11y")          # built at runtime: this file must not trip the check itself
    _put(root, "data.json", b"0" * (6 * 1024 * 1024))
    r = check(root)
    assert r["block"] == []
    warned = {rel for rel, _ in r["warn"]}
    assert "docs/log.md" in warned and "data.json" in warned and "LICENSE" in warned
    assert main([str(root)]) == 0


def test_the_five_examples_are_counted(tmp_path):
    root = _repo(tmp_path)
    for n in ("a", "b", "c", "d", "e"):
        _put(root, f"demo_examples/{n}/video.mp4")
    assert not [w for w in check(root)["warn"] if w[0] == "demo_examples/"]
    shutil.rmtree(root / "demo_examples" / "e")
    assert [w for w in check(root)["warn"] if w[0] == "demo_examples/"]


def test_a_folder_that_is_not_a_repository_gets_a_clear_message(tmp_path, capsys=None):
    assert "not a git repository" in check(tmp_path)["error"] and main([str(tmp_path)]) == 2