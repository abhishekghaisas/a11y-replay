"""Write report.json and a self-contained, accessible report.html for an AuditReport.

The HTML embeds its evidence images, so one file can be emailed or attached to a ticket. A tool that
audits accessibility should not ship an inaccessible report: the page uses semantic headings, tables with
header cells, alt text on every image, status shown in words (never by color alone), visible keyboard
focus, and light/dark color schemes.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1d2733;--muted:#4a5565;--card:#f4f6f8;--line:#c9d1da;--bad:#9b1c1c;--warn:#7a4b00;--ok:#14633a;--link:#0b57d0}
@media (prefers-color-scheme:dark){:root{--bg:#10161d;--fg:#e8edf2;--muted:#b4bfcc;--card:#1a232d;--line:#3a4655;--bad:#ff9d9d;--warn:#ffcf70;--ok:#7fe0a6;--link:#9ec5ff}}
body{font:16px/1.55 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0}
main{max-width:920px;margin:0 auto;padding:24px 20px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:21px;margin:36px 0 12px;border-bottom:2px solid var(--line);padding-bottom:6px}h3{font-size:18px;margin:0 0 6px}
.note{background:var(--card);border-left:6px solid var(--warn);padding:12px 16px;border-radius:4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px 18px;margin:14px 0}
.meta{color:var(--muted);margin:0 0 8px}.tag{display:inline-block;border:1px solid currentColor;border-radius:4px;padding:0 8px;font-size:14px;margin-right:6px}
.finding .tag.s{color:var(--bad)}.review .tag.s{color:var(--warn)}.pass .tag.s{color:var(--ok)}
figure{margin:10px 0 0}figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}figcaption{color:var(--muted);font-size:14px}
table{border-collapse:collapse;width:100%}th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}th{background:var(--card)}
a:focus-visible,summary:focus-visible{outline:3px solid var(--link);outline-offset:2px}
""".strip()

STATUS_WORDS = {"ran": "Ran", "skipped": "Skipped", "not_implemented": "Not implemented"}


def _t(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def _img(out: Path, rel: str, alt: str) -> str:
    p = out / rel
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(alt, quote=True)}" src="data:image/png;base64,{b64}">'


def _item(out: Path, it) -> str:
    cls = {"finding": "finding", "pass": "pass"}.get(it.status, "review")
    label = {"finding": "Finding", "pass": "Judged fine"}.get(it.status, "Needs review")
    who = "a language model (no calibrated confidence)" if it.method == "vlm" else "the classical detector"
    figs = "".join(
        f"<figure>{_img(out, e['file'], e['caption'] + '. ' + it.title)}<figcaption>{html.escape(e['caption'])}</figcaption></figure>"
        for e in it.evidence)
    span = _t(it.start) if it.end <= it.start else f"{_t(it.start)} to {_t(it.end)}"
    return (f'<article class="card {cls}" aria-labelledby="{html.escape(it.id)}">'
            f'<h3 id="{html.escape(it.id)}">{html.escape(it.title)}</h3>'
            f'<p class="meta"><span class="tag s">{label}</span><span class="tag">WCAG {html.escape(it.criterion)}</span>'
            f'<span class="tag">at {span}</span> decided by {who}</p>'
            f"<p>{html.escape(it.reasoning)}</p>{figs}</article>")


def render_html(report, out_dir) -> str:
    out = Path(out_dir)
    s, f, r = report.sync, report.findings, report.needs_review
    stat = report.stats
    cost = stat.get("est_cost_usd")
    model_line = (f"Model: {html.escape(str(stat['model']))}; {stat.get('vlm_calls', 0)} call(s)"
                  + (f"; about ${cost:.4f} (estimate)" if cost is not None else "")) if stat.get("model") else "No language model was used."
    def _row(c):
        name, detail = html.escape(c["name"]), html.escape(c["detail"])
        status = STATUS_WORDS.get(c["status"], c["status"])
        return f'<tr><th scope="row">{name}</th><td>{status}</td><td>{detail}</td></tr>'

    rows = "".join(_row(c) for c in report.checks)
    ran = [c for c in report.checks if c["status"] == "ran"]
    skipped = [c for c in report.checks if c["status"] == "skipped"]
    if skipped:
        why = "".join(f"<li>{html.escape(c['name'])}: {html.escape(c['detail'])}</li>" for c in skipped)
        coverage = (f'<p class="note" role="note"><strong>Only {len(ran)} of {len(ran) + len(skipped)} available checks ran, so '
                    f"a result of zero findings does not mean the recording is clean.</strong> Not run:</p><ul>{why}</ul>")
    else:
        coverage = f"<p>All {len(ran)} available checks ran.</p>"
    sync_cls = "" if s["trusted"] else " <strong>Not trusted.</strong>"
    body = [
        f"<h1>Accessibility audit: {html.escape(report.recording)}</h1>",
        f'<p class="meta">Generated {html.escape(report.created)} by a11y-replay {html.escape(report.version)}. '
        f"Recording length {stat.get('video_seconds', '?')} s.</p>",
        '<p class="note" role="note"><strong>This is a screening aid, not a compliance test.</strong> '
        "It checks a few WCAG 2.1 criteria from a screen recording. A clean report does not mean the page is "
        "accessible, and every finding should be confirmed by a person.</p>",
        "".join(f'<p class="note" role="alert"><strong>Warning.</strong> {html.escape(w)}</p>' for w in report.warnings),
        '<h2 id="summary">Summary</h2>',
        f"<p><strong>{len(f)}</strong> finding(s) and <strong>{len(r)}</strong> item(s) needing review.</p>",
        coverage,
        f"<p>Clock sync between video and key log: offset {s['offset_s']:+.3f} s from {s['pairs']} key/change pair(s) "
        f"({s.get('calibration_pairs', 0)} calibration flashes), spread "
        f"{'n/a' if s.get('spread_s') is None else format(s['spread_s'], '.2f') + ' s'}.{sync_cls} "
        f"{html.escape(s['note'])}</p>",
        f"<p>{model_line}</p>",
        '<h2 id="findings">Findings</h2>',
        "".join(_item(out, i) for i in f) or "<p>None.</p>",
        '<h2 id="review">Needs review</h2>',
        "<p>The tool could not decide these. A person (or a model, with <code>--model</code>) should look.</p>",
        "".join(_item(out, i) for i in r) or "<p>None.</p>",
        (f'<h2 id="passes">Judged fine ({len(report.passes)})</h2>'
         "<p>Shown because the audit was run with <code>--show-passes</code>. Look at these as critically as the findings: "
         "the tool can be wrong in both directions.</p>" + "".join(_item(out, i) for i in report.passes)) if report.passes else "",
        '<h2 id="checks">Checks</h2>',
        f'<table><caption>What was and was not checked</caption><thead><tr><th scope="col">Check</th>'
        f'<th scope="col">Status</th><th scope="col">Detail</th></tr></thead><tbody>{rows}</tbody></table>',
        '<h2 id="limits">What this report does not cover</h2>',
        "<ul><li>Any WCAG criterion not listed above, and anything that needs human judgment.</li>"
        "<li>Problems that only exist in the page's code (a missing label, a wrong role): this reads pixels and "
        "speech, not the accessibility tree.</li>"
        "<li>Findings from a language model carry no calibrated confidence.</li></ul>",
    ]
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Accessibility audit: {html.escape(report.recording)}</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(body)}</main></body></html>")


def write_report(report, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    path = out / "report.html"
    path.write_text(render_html(report, out), encoding="utf-8")
    return path