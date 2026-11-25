"""Markdown + JSON report writers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from .models import AuditResult, Notebook, render_cell

VERDICT_ICON = {"SUPPORTED": "✅", "CONTRADICTED": "❌", "UNVERIFIABLE": "❔", "METHODOLOGICAL_RISK": "⚠️"}
SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡", "info": "🔵"}


def _esc(text: str, limit: Optional[int] = None) -> str:
    text = " ".join(str(text).split())
    if limit and len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("|", "\\|")


def _cell_link(i: int) -> str:
    return f"[Cell {i}](#cell-{i})"


def _cells_links(cells: Sequence[int]) -> str:
    return ", ".join(_cell_link(i) for i in sorted(set(cells))) or "—"


def render_markdown(result: AuditResult, nb: Optional[Notebook] = None) -> str:
    s = result.summary()
    lines: List[str] = []
    add = lines.append
    add("# Notebook Audit Report")
    add("")
    add(f"- **Notebook:** `{result.notebook_path}`")
    add(f"- **Artifacts:** {', '.join(f'`{p}`' for p in result.artifact_paths) or '_notebook markdown cells_'}")
    add(f"- **Generated:** {result.generated_at}")
    add(f"- **Mode:** `{result.mode}`")
    ns = result.notebook_summary
    add(f"- **Notebook:** {ns.get('cells')} cells ({ns.get('code_cells')} code, {ns.get('executed_code_cells')} executed, {ns.get('images')} images)")
    add("")
    add("> Cell numbers are 1-based positions in the notebook (Cell 1 = first cell, markdown included).")
    add("")

    # ---------------------------------------------------------------- findings & gaps
    if result.findings:
        _render_findings(result, add)
    if result.gaps:
        _render_gaps(result, add)

    # ---------------------------------------------------------------- summary
    add("## Summary")
    add("")
    add("| Verdict | Count |")
    add("|---|---|")
    for v, n in s["verdicts"].items():
        add(f"| {VERDICT_ICON[v]} {v} | {n} |")
    add("")
    sev = s["risks_by_severity"]
    add("**Findings by severity:** " + " · ".join(f"{SEV_ICON[k]} {k}: {v}" for k, v in sev.items()))
    add("")
    headline = [r for r in result.risks if r.severity == "high"]
    if headline:
        add("### Top issues")
        add("")
        for r in headline[:8]:
            add(f"- {SEV_ICON[r.severity]} **{r.finding_id}** — {r.message} ({_cells_links(r.cell_indices)})")
        add("")

    # ---------------------------------------------------------------- claims table
    add("## Claim Verdicts")
    add("")
    if not result.claims:
        add("_No written claims were checked. Add slides, a paper draft or notes (or text cells in the notebook) to check them against the outputs._")
        add("")
    add("| ID | Claim | Category | Verdict | Strength | Evidence cells |")
    add("|---|---|---|---|---|---|")
    for c in result.claims:
        v = result.verdict_for(c.claim_id)
        if v is None:
            continue
        cells = sorted({e.cell_index for e in v.evidence})
        add(f"| [{c.claim_id}](#{c.claim_id.lower()}) | {_esc(c.text, 110)} | {c.category} | {VERDICT_ICON[v.verdict]} {v.verdict} | {v.strength or ''}{'' if v.traced else ' (untraced)' if v.strength == 'MEASURED' else ''} | {_cells_links(cells)} |")
    add("")

    # ---------------------------------------------------------------- findings by engine
    sections = [
        ("evidence", "Evidence & Discrepancy Findings"),
        ("risk", "Methodological Risks"),
        ("figure", "Figure & Visual Audit"),
        ("strength", "Epistemic Modesty"),
    ]
    for engine, title in sections:
        rs = [r for r in result.risks if r.engine == engine]
        add(f"## {title}")
        add("")
        if not rs:
            add("_No findings._")
            add("")
            continue
        for r in rs:
            add(f"### {SEV_ICON[r.severity]} {r.finding_id} · `{r.check}` · {r.severity}")
            add("")
            add(r.message)
            add("")
            if r.detail:
                add(f"> {r.detail}")
                add("")
            meta = []
            if r.cell_indices:
                meta.append(f"Cells: {_cells_links(r.cell_indices)}")
            if r.claim_ids:
                meta.append("Claims: " + ", ".join(f"[{c}](#{c.lower()})" for c in r.claim_ids))
            if meta:
                add(" · ".join(meta))
                add("")

    # ---------------------------------------------------------------- per-claim detail
    add("## Claim Details")
    add("")
    plans = {p.claim_id: p for p in result.plans}
    execs = {e.claim_id: e for e in result.executions}
    for c in result.claims:
        v = result.verdict_for(c.claim_id)
        if v is None:
            continue
        add(f'<a id="{c.claim_id.lower()}"></a>')
        add(f"### {c.claim_id} — {VERDICT_ICON[v.verdict]} {v.verdict} (confidence {v.confidence:.2f})")
        add("")
        add(f"> {_esc(c.text)}")
        add("")
        if c.original_text.strip() != c.text.strip():
            add(f"- **Original:** {_esc(c.original_text, 300)}")
        add(f"- **Source:** `{c.source}`" + (f" · context: _{_esc(c.context)}_" if c.context else ""))
        add(f"- **Category:** {c.category}" + (f" · tags: {', '.join(c.tags)}" if c.tags else "") + (f" · direction: {c.direction}" if c.direction else ""))
        add(f"- **Strength:** {v.strength}{'' if v.traced else ' — not traced to any output'}")
        p = plans.get(c.claim_id)
        if p:
            add(f"- **Plan ({p.planner}):** {_cells_links(p.target_cells)} — {_esc(p.rationale, 300)}")
        add(f"- **Verifier ({v.verifier}):** {v.rationale}")
        if v.rules_fired:
            add(f"- **Rules fired:** {', '.join(f'`{r}`' for r in v.rules_fired)}")
        ex = execs.get(c.claim_id)
        if ex and ex.number_checks:
            add("- **Number checks:**")
            for nc in ex.number_checks:
                add(f"  - `{nc.raw}` → **{nc.status}** — {nc.note}")
        if v.evidence:
            add("- **Evidence:**")
            for e in v.evidence[:6]:
                lab = f" _(label: {e.label})_" if e.label else ""
                add(f"  - {_cell_link(e.cell_index)} {e.location} [{e.relation}]: `{_esc(e.snippet, 160)}`{lab}")
        if v.risk_ids:
            add(f"- **Related findings:** {', '.join(v.risk_ids)}")
        if v.does_not_prove:
            add(f"- **What this does not prove:** {v.does_not_prove}")
        add("")

    # ---------------------------------------------------------------- appendix
    referenced: Set[int] = set()
    for v in result.verdicts:
        referenced |= {e.cell_index for e in v.evidence}
    for p in result.plans:
        referenced |= set(p.target_cells)
    for r in result.risks:
        referenced |= set(r.cell_indices)
    for f in result.findings:
        referenced |= set(f.cells)
    for g in result.gaps:
        referenced |= set(g.cells)
    if nb is not None and referenced:
        add("## Evidence Appendix")
        add("")
        for i in sorted(referenced):
            cell = nb.get(i)
            if cell is None:
                continue
            add(f'<a id="cell-{i}"></a>')
            exec_tag = f" · In [{cell.execution_count}]" if cell.execution_count is not None else ""
            add(f"### Cell {i} ({cell.cell_type}{exec_tag})")
            add("")
            add("```python" if cell.cell_type == "code" else "```markdown")
            add(cell.source_code.rstrip()[:4000])
            add("```")
            out = cell.output_text
            if out.strip():
                add("")
                add("Output:")
                add("")
                add("```text")
                add(out.rstrip()[:3000] + ("\n… [truncated]" if len(out) > 3000 else ""))
                add("```")
            add("")

    if result.warnings:
        add("## Warnings")
        add("")
        for w in result.warnings:
            add(f"- {w}")
        add("")
    return "\n".join(lines)


TRUST_LABEL = {
    "solid": "✅ Solid",
    "at_risk": "⚠️ At risk",
    "unconfirmed": "❔ Unconfirmed",
    "visual": "👁️ Read from figure",
}
BASIS_LABEL = {
    "MEASURED": "printed by the notebook",
    "DERIVED": "worked out from printed values",
    "CODE": "typed into the plotting code",
    "VISUAL": "read from a figure by Claude",
    "INTERPRETATION": "an interpretation by Claude",
}


def _render_findings(result: AuditResult, add) -> None:
    by_trust = result.summary()["findings_by_trust"]
    add("## What your notebook found")
    add("")
    drafted = "Claude and the offline checks" if any(f.source == "llm" for f in result.findings) else "the saved outputs"
    add(f"_Drafted automatically from {drafted}. Every finding links to its cell; **Trust** says how far to rely on it._ "
        + " · ".join(f"{TRUST_LABEL[t]}: {n}" for t, n in by_trust.items() if n))
    add("")
    add("| # | Finding | Where | Trust |")
    add("|---|---|---|---|")
    for i, f in enumerate(result.findings, start=1):
        add(f"| [{i}](#{f.finding_id.lower()}) | {_esc(f.text, 160)} | {_cells_links(f.cells)} | {TRUST_LABEL.get(f.trust, f.trust)} |")
    add("")
    for i, f in enumerate(result.findings, start=1):
        add(f'<a id="{f.finding_id.lower()}"></a>')
        add(f"**{i}. {_esc(f.text)}**  ")
        add(f"{TRUST_LABEL.get(f.trust, f.trust)} · {BASIS_LABEL.get(f.basis, f.basis)} · {_cells_links(f.cells)}")
        shown = set()
        for e in f.evidence[:3]:
            if e.snippet and (e.cell_index, e.snippet) not in shown:
                shown.add((e.cell_index, e.snippet))
                add(f"- Evidence ({_cell_link(e.cell_index)} {e.location}): `{_esc(e.snippet, 160)}`")
        for c in f.caveats:
            add(f"- ⚑ {c}")
        add("")


def _render_gaps(result: AuditResult, add) -> None:
    add("## What's missing")
    add("")
    add("_Next steps that would make the results more trustworthy, most important first._")
    add("")
    for i, g in enumerate(result.gaps, start=1):
        where = f" ({_cells_links(g.cells)})" if g.cells else ""
        add(f"### {i}. {g.title} {SEV_ICON.get(g.severity, '')}")
        add("")
        add(f"{g.why}{where}")
        add("")
        if "\n" in g.how or "(" in g.how and "=" in g.how:
            add("```python")
            add(g.how)
            add("```")
        else:
            add(g.how)
        add("")


def write_reports(result: AuditResult, output_dir: str, nb: Optional[Notebook] = None, formats: Sequence[str] = ("md", "json")) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    if "md" in formats:
        p = out / "audit_report.md"
        p.write_text(render_markdown(result, nb), encoding="utf-8")
        paths["md"] = str(p)
    if "json" in formats:
        p = out / "audit_report.json"
        p.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        paths["json"] = str(p)
    return paths
