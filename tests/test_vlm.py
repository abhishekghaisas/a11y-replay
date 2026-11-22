"""VLM layer tests. No network: the SDK is replaced by a fake object."""
import base64
import json
import types

import numpy as np

from a11y_replay.vlm.client import AnthropicClient, CachedClient, MockClient, OllamaClient
from a11y_replay.vlm.pricing import estimate_cost
from a11y_replay.vlm.prompts import FOCUS_JUDGE_INTRO, FOCUS_JUDGE_QUESTION
from a11y_replay.vlm.tasks import build_focus_content, crop_region, judge_focus, parse_judge_json

OK = '{"indicator_visible": "yes", "indicator_type": "outline", "evidence": "orange ring"}'
STYLE_NAMES = ["strong_ring", "box_shadow_ring", "thin_dotted", "underline", "bg_change",
               "faint_ring", "selection_only", "subtle_color"]


def _img(v=200, h=120, w=300):
    return np.full((h, w, 3), v, np.uint8)


def test_parse_plain_fenced_and_wrapped_json():
    assert parse_judge_json(OK)["answer"] == "yes"
    assert parse_judge_json("```json\n" + OK + "\n```")["answer"] == "yes"
    assert parse_judge_json("Sure! " + OK + " Hope that helps.")["answer"] == "yes"
    assert parse_judge_json(OK.replace("yes", "YES"))["answer"] == "yes"


def test_parse_rejects_bad_output():
    assert parse_judge_json("") is None
    assert parse_judge_json("not json") is None
    assert parse_judge_json('{"indicator_visible": "maybe"}') is None
    assert parse_judge_json('{"something": "else"}') is None


def test_prompt_does_not_leak_the_benchmark():
    text = FOCUS_JUDGE_INTRO + FOCUS_JUDGE_QUESTION
    assert not any(s in text for s in STYLE_NAMES)
    content = build_focus_content(b"a", b"b")
    assert [c["type"] for c in content].count("image") == 2


def test_crop_pads_clamps_and_downscales():
    f = np.zeros((720, 1280, 3), np.uint8)
    c = crop_region(f, (100, 100, 50, 40))
    assert c.shape[:2] == (160, 170)                         # 40+120 tall, 50+120 wide
    assert crop_region(f, (0, 0, 10, 10)).shape[:2] == (70, 70)  # clamped at the top-left corner
    assert max(crop_region(f, None).shape[:2]) <= 900            # whole frame is downscaled


def test_judge_focus_happy_path_counts_one_call():
    client = MockClient([OK])
    r = judge_focus(client, _img(), _img(), (10, 10, 50, 30))
    assert r.answer == "yes" and r.parse_ok and r.attempts == 1 and client.meter.calls == 1


def test_judge_focus_retries_once_on_bad_json():
    client = MockClient(["I think yes", OK])
    r = judge_focus(client, _img(), _img())
    assert r.answer == "yes" and r.attempts == 2 and client.meter.calls == 2
    assert client.requests[1][-1]["text"].startswith("Reply with ONLY")


def test_judge_focus_gives_up_as_unsure_never_guesses():
    client = MockClient(["nope", "still nope"])
    r = judge_focus(client, _img(), _img())
    assert r.answer == "unsure" and not r.parse_ok and r.attempts == 2


class FakeSDK:
    """Mimics anthropic.Anthropic().messages.create with SDK-style response objects."""

    def __init__(self, text=OK):
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create)
        self._text = text

    def _create(self, **kw):
        self.calls.append(kw)
        block_t = types.SimpleNamespace(type="thinking", thinking="...")
        block_x = types.SimpleNamespace(type="text", text=self._text)
        return types.SimpleNamespace(content=[block_t, block_x],
                                     usage=types.SimpleNamespace(input_tokens=321, output_tokens=45))


def test_anthropic_request_follows_sonnet_5_5_rules():
    sdk = FakeSDK()
    client = AnthropicClient(sdk_client=sdk)
    out = client.complete(build_focus_content(b"\x89PNG-before", b"\x89PNG-after"))
    kw = sdk.calls[0]
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["extra_body"]["thinking"] == {"type": "between_tools"}   # "disabled" would be a 400
    assert kw["extra_body"]["output_config"] == {"effort": "medium"}
    for banned in ("temperature", "top_p", "top_k", "tool_choice"):    # 400s on Sonnet 5.5
        assert banned not in kw and banned not in kw["extra_body"]
    imgs = [b for b in kw["messages"][0]["content"] if b["type"] == "image"]
    assert len(imgs) == 2 and imgs[0]["source"]["media_type"] == "image/png"
    assert base64.b64decode(imgs[0]["source"]["data"]) == b"\x89PNG-before"
    assert out.text == OK and out.input_tokens == 321 and out.output_tokens == 45  # thinking block skipped
    assert client.meter.calls == 1


def test_anthropic_adaptive_thinking_omits_thinking_field_and_raises_token_budget():
    client = AnthropicClient(thinking="adaptive", effort="high", sdk_client=FakeSDK())
    req = client.build_request(build_focus_content(b"a", b"b"))
    assert "thinking" not in req.get("extra_body", {}) and req["max_tokens"] >= 4000
    assert req["extra_body"]["output_config"] == {"effort": "high"}


def test_other_models_get_no_sonnet_specific_fields():
    req = AnthropicClient(model="claude-haiku-5-5", sdk_client=FakeSDK()).build_request(
        build_focus_content(b"a", b"b"))
    assert "extra_body" not in req


def test_ollama_payload_shape():
    p = OllamaClient("qwen3-vl:8b").build_payload(build_focus_content(b"a", b"b"))
    assert p["model"] == "qwen3-vl:8b" and p["stream"] is False and len(p["messages"][0]["images"]) == 2


def test_cached_client_replays_without_calling_again(tmp_path):
    inner = MockClient([OK])
    c = CachedClient(inner, tmp_path)
    content = build_focus_content(b"a", b"b")
    first, second = c.complete(content), c.complete(content)
    assert first.text == second.text and inner.meter.calls == 1 and c.meter.cache_hits == 1
    assert c.complete(build_focus_content(b"a", b"DIFFERENT")).text == OK and inner.meter.calls == 2


def test_cost_estimates():
    assert abs(estimate_cost("claude-sonnet-5-5", 1_000_000, 1_000_000) - 12.0) < 1e-9
    assert estimate_cost("ollama:qwen3-vl:8b", 5, 5) == 0.0
    assert estimate_cost("some-unknown-model", 5, 5) is None