"""Model-agnostic VLM clients. Every call is counted (VLM calls per minute of video is a headline metric).

Content is provider-neutral: a list of {"type": "text", "text": str} and {"type": "image", "png": bytes}.

  AnthropicClient : Claude via the Anthropic SDK. Request details below follow Anthropic's docs for
                    Sonnet 5.5 as read on 2026-11-10; the live API has NOT been called from the
                    development environment, only a fake SDK object.
  OllamaClient    : a local open model (e.g. qwen3-vl:8b) through Ollama's HTTP API. Untested against
                    a real server.
  MockClient      : scripted answers for tests.
  CachedClient    : wraps any client, replaying saved responses so evaluation reruns are free and
                    repeatable (these models cannot be made deterministic with temperature, because
                    Sonnet 5.5 rejects non-default sampling settings).

Sonnet 5.5 notes (from the docs): thinking cannot be disabled; send {"type": "between_tools"} to turn
off up-front thinking (allowed at low/medium/high effort); effort goes in output_config; forced tool
use and non-default temperature/top_p/top_k return 400, so none of those are sent.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SONNET = "claude-sonnet-5-5"


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""


@dataclass
class UsageMeter:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cache_hits: int = 0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.latency_s += r.latency_s


class VLMClient:
    model: str = "unknown"

    def __init__(self) -> None:
        self.meter = UsageMeter()

    def signature(self) -> str:
        """Everything that changes the answer; used as part of the cache key."""
        return self.model

    def complete(self, content: list, max_tokens: int = 600) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


def _get(obj, name, default=None):
    """Read a field from an SDK object or a plain dict."""
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


class AnthropicClient(VLMClient):
    def __init__(self, model: str = SONNET, effort: str | None = None, thinking: str | None = None,
                 max_tokens: int | None = None, sdk_client=None):
        """effort: low|medium|high|xhigh|max (Sonnet 5.5). thinking: "between_tools", "adaptive" or None.
        For Sonnet 5.5 the defaults are between_tools + medium effort (a starting point to be tuned)."""
        super().__init__()
        self.model = model
        sonnet = model == SONNET
        self.effort = effort if effort is not None else ("medium" if sonnet else None)
        self.thinking = thinking if thinking is not None else ("between_tools" if sonnet else None)
        self.max_tokens = max_tokens
        self._sdk = sdk_client

    def signature(self) -> str:
        return f"{self.model}|effort={self.effort}|thinking={self.thinking}"

    def _sdk_client(self):
        if self._sdk is None:
            try:
                import anthropic  # noqa: PLC0415 (optional dependency)
            except ImportError as e:
                raise RuntimeError("Install the Anthropic SDK first:  pip install anthropic") from e
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in .env; see .env.example).")
            self._sdk = anthropic.Anthropic()
        return self._sdk

    def build_request(self, content: list, max_tokens: int | None = None) -> dict:
        blocks = []
        for c in content:
            if c["type"] == "text":
                blocks.append({"type": "text", "text": c["text"]})
            else:
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.b64encode(c["png"]).decode("ascii")}})
        # Fields the SDK may not know yet travel in extra_body, so this works across SDK versions.
        extra = {}
        if self.thinking == "between_tools":
            extra["thinking"] = {"type": "between_tools"}
        if self.effort:
            extra["output_config"] = {"effort": self.effort}
        budget = max_tokens or self.max_tokens or (4000 if self.thinking != "between_tools" else 600)
        req = {"model": self.model, "max_tokens": budget,
               "messages": [{"role": "user", "content": blocks}]}
        if extra:
            req["extra_body"] = extra
        return req

    def complete(self, content: list, max_tokens: int = 600) -> LLMResponse:
        req = self.build_request(content, self.max_tokens or max_tokens)
        t0 = time.monotonic()
        resp = self._sdk_client().messages.create(**req)
        dt = time.monotonic() - t0
        text = "".join(_get(b, "text", "") or "" for b in _get(resp, "content", [])
                       if _get(b, "type") == "text")
        usage = _get(resp, "usage")
        out = LLMResponse(text=text, input_tokens=int(_get(usage, "input_tokens", 0) or 0),
                          output_tokens=int(_get(usage, "output_tokens", 0) or 0),
                          latency_s=dt, model=self.model)
        self.meter.add(out)
        return out


class OllamaClient(VLMClient):
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        super().__init__()
        self.model = f"ollama:{model}"
        self._name, self._host = model, host.rstrip("/")

    def build_payload(self, content: list) -> dict:
        text = "\n".join(c["text"] for c in content if c["type"] == "text")
        images = [base64.b64encode(c["png"]).decode("ascii") for c in content if c["type"] == "image"]
        return {"model": self._name, "stream": False, "format": "json",
                "options": {"temperature": 0},
                "messages": [{"role": "user", "content": text, "images": images}]}

    def complete(self, content: list, max_tokens: int = 600) -> LLMResponse:
        body = json.dumps(self.build_payload(content)).encode()
        req = urllib.request.Request(f"{self._host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        out = LLMResponse(text=data.get("message", {}).get("content", ""),
                          input_tokens=int(data.get("prompt_eval_count", 0)),
                          output_tokens=int(data.get("eval_count", 0)),
                          latency_s=time.monotonic() - t0, model=self.model)
        self.meter.add(out)
        return out


class MockClient(VLMClient):
    """Scripted client for tests. `responder(content) -> str` (or a list of strings, used in order)."""

    def __init__(self, responder, model: str = "mock"):
        super().__init__()
        self.model = model
        self._responder = responder
        self.requests: list = []

    def complete(self, content: list, max_tokens: int = 600) -> LLMResponse:
        self.requests.append(content)
        if callable(self._responder):
            text = self._responder(content)
        else:
            text = self._responder[min(len(self.requests) - 1, len(self._responder) - 1)]
        out = LLMResponse(text=text, input_tokens=100, output_tokens=40, latency_s=0.0, model=self.model)
        self.meter.add(out)
        return out


class CachedClient(VLMClient):
    """Replays saved responses; only calls the wrapped client on a cache miss."""

    def __init__(self, inner: VLMClient, cache_dir: str | Path):
        super().__init__()
        self.inner, self.model = inner, inner.model
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def signature(self) -> str:
        return self.inner.signature()

    def _key(self, content: list) -> str:
        h = hashlib.sha256(self.inner.signature().encode())
        for c in content:
            h.update(c["text"].encode() if c["type"] == "text" else hashlib.sha256(c["png"]).digest())
        return h.hexdigest()

    def complete(self, content: list, max_tokens: int = 600) -> LLMResponse:
        path = self.dir / f"{self._key(content)}.json"
        if path.exists():
            d = json.loads(path.read_text())
            out = LLMResponse(**d["response"])
            self.meter.cache_hits += 1
            self.meter.add(out)
            return out
        out = self.inner.complete(content, max_tokens)
        self.meter.add(out)
        path.write_text(json.dumps({"signature": self.inner.signature(), "response": out.__dict__}, indent=1))
        return out