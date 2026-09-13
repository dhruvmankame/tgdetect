#!/usr/bin/env python3
"""Inject the over-smoothing sweep results into the parity-audit HTML.

Parses the raw stdout of `scripts/diagnose_oversmoothing.py` and replaces the
<!-- SWEEP_TABLE --> marker in PROMETHEUS_PAPER_PARITY_AUDIT.html, so the
numbers in the report can never drift from the run that produced them.

    python scripts/inject_sweep_table.py <sweep_output.txt>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HTML = Path(__file__).resolve().parent.parent / "PROMETHEUS_PAPER_PARITY_AUDIT.html"
MARKER = "<!-- SWEEP_TABLE -->"


def parse(text: str) -> list[dict]:
    blocks: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        m = re.match(r"\s*--- avg degree (\d+) \(edges/window ~ (\d+)\) ---", line)
        if m:
            cur = {"degree": int(m.group(1)), "edges": int(m.group(2)),
                   "ceiling": None, "rows": []}
            blocks.append(cur)
            continue
        m = re.match(r"\s*logistic-regression ceiling on raw features: "
                     r"AUC-ROC ([\d.]+)\s+AUC-PR ([\d.]+)", line)
        if m and cur is not None:
            cur["ceiling"] = (float(m.group(1)), float(m.group(2)))
            continue
        m = re.match(r"\s*(\d+)\s+(True|False)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%", line)
        if m and cur is not None:
            cur["rows"].append({
                "layers": int(m.group(1)), "residual": m.group(2) == "True",
                "roc": float(m.group(3)), "pr": float(m.group(4)),
                "distinct": float(m.group(5)),
            })
    return blocks


def cell(v: float, good: float = 0.85, bad: float = 0.65) -> str:
    cls = "v-green" if v >= good else ("v-red" if v <= bad else "v-yellow")
    return f'<td class="num {cls}">{v:.4f}</td>'


def render(blocks: list[dict]) -> str:
    out = ['<div class="tw">', "<table>", "<thead><tr>",
           "<th>Avg degree</th><th>Edges/window</th><th>Layers</th><th>Residual</th>",
           '<th class="num">AUC-ROC</th><th class="num">AUC-PR</th>',
           '<th class="num">Distinct preds</th><th>Note</th>',
           "</tr></thead>", "<tbody>"]
    for b in blocks:
        n = len(b["rows"])
        for i, r in enumerate(b["rows"]):
            paper = r["layers"] == 3 and not r["residual"]
            tr = ' class="paper-row"' if paper else ""
            out.append(f"<tr{tr}>")
            if i == 0:
                out.append(f'<td rowspan="{n}" class="num"><strong>{b["degree"]}</strong></td>')
                out.append(f'<td rowspan="{n}" class="num">~{b["edges"]:,}</td>')
            out.append(f'<td class="num">{r["layers"]}</td>')
            out.append(f'<td>{"yes" if r["residual"] else "no"}</td>')
            out.append(cell(r["roc"]))
            out.append(cell(r["pr"], good=0.70, bad=0.40))
            out.append(f'<td class="num">{r["distinct"]:.1f}%</td>')
            note = ""
            if paper:
                note = '<span class="b b-gap">paper config</span>'
            elif r["layers"] == 3 and r["residual"]:
                note = '<span class="b b-ok">+residual</span>'
            out.append(f"<td>{note}</td>")
            out.append("</tr>")
        if b["ceiling"]:
            out.append(
                f'<tr style="background:rgba(88,166,255,.07)">'
                f'<td colspan="4" style="color:var(--accent)">'
                f'&nbsp;&nbsp;↳ logistic regression on the raw features (ceiling)</td>'
                f'<td class="num v-green">{b["ceiling"][0]:.4f}</td>'
                f'<td class="num v-green">{b["ceiling"][1]:.4f}</td>'
                f'<td class="num">—</td><td></td></tr>')
    out += ["</tbody>", "</table>", "</div>"]
    return "\n".join(out)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    blocks = parse(Path(sys.argv[1]).read_text())
    if not blocks:
        raise SystemExit("no sweep blocks parsed — is that the right file?")
    html = HTML.read_text()
    if MARKER not in html:
        raise SystemExit(f"{MARKER} not found in {HTML}")
    HTML.write_text(html.replace(MARKER, render(blocks)))
    rows = sum(len(b["rows"]) for b in blocks)
    print(f"injected {len(blocks)} density blocks / {rows} rows into {HTML.name}")


if __name__ == "__main__":
    main()
