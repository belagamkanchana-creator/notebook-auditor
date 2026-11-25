"""Evidence mapping & discrepancy auditor.

* Turns every printed number in the notebook into a labelled :class:`Observation`.
* Checks each number in a claim against those observations (match / derived / mismatch).
* Produces the claim -> cell evidence map and discrepancy findings for the report.
"""
from __future__ import annotations

import re
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

from ..code_analysis import CodeIndex
from ..models import (
    Cell,
    Claim,
    ClaimVerdict,
    EvidenceItem,
    ExecutionEvidence,
    NumberCheck,
    NumberMention,
    Notebook,
    Observation,
    RiskFinding,
    SeriesEvidence,
)
from ..textnum import (
    BOUNDED_KINDS,
    comparable,
    content_tokens,
    extract_numbers,
    fmt_value,
    kinds_of_text,
    nearest_kinds,
    qualifiers_of_tokens,
    series_direction,
    tokenize,
    values_match,
)

BROAD_KINDS = {"count", "features", "classes", "epochs"}
_LETTERS_RE = re.compile(r"[A-Za-z]")


# --------------------------------------------------------------------------- observations
def _last_code_line(src: str) -> str:
    for line in reversed(src.splitlines()):
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return ""


def _print_lines(src: str) -> List[str]:
    return [l.strip() for l in src.splitlines() if re.search(r"\bprint\s*\(", l)]


def _mk_obs(cell: Cell, m: NumberMention, label: str, line: str, location: str = "output", kinds=None, quals=None, row_key: str = "") -> Observation:
    if kinds is None:
        kinds = set(m.kinds)
    if quals is None:
        quals = set(m.qualifiers)
    kinds = set(kinds)
    if kinds & BOUNDED_KINDS and not _plausible_bounded(m):
        kinds -= BOUNDED_KINDS  # "accuracy ... 1692" is a support count, not an accuracy
    return Observation(
        cell_index=cell.cell_index,
        location=location,
        value=m.value,
        raw=m.raw,
        is_percent=m.is_percent,
        decimals=m.decimals,
        label=" ".join(label.split())[:120],
        kinds=sorted(kinds),
        qualifiers=sorted(quals),
        line=line.strip()[:200],
        row_key=row_key,
    )


def _plausible_bounded(m: NumberMention) -> bool:
    """Accuracy/F1/... are in [0, 1], or a percentage."""
    return m.is_percent or 0 <= m.value <= 1 or (1 < m.value <= 100 and m.decimals >= 1)


def _table_observations(cell: Cell, table: List[List[str]]) -> List[Observation]:
    if len(table) < 2:
        return []
    header = table[0]
    obs: List[Observation] = []
    for row in table[1:]:
        if not row:
            continue
        # Name the row by its first text cell: a DataFrame's index is often just 0, 1, 2 while a "model" column says what the row is.
        name_j = next((j for j, v in enumerate(row) if re.search(r"[A-Za-z]{2,}", v) and v.lower() != "nan"), 0)
        row_label = row[name_j]
        offset = len(header) - len(row)
        for j, value in enumerate(row[1:], start=1):
            if j == name_j:
                continue
            hj = j + offset if 0 <= j + offset < len(header) else None
            col = header[hj] if hj is not None else ""
            for m in extract_numbers(value):
                label = f"{row_label} {col}"
                toks = tokenize(label)
                obs.append(_mk_obs(cell, m, label, f"{row_label} | {col} = {value}", kinds=kinds_of_text(label), quals=qualifiers_of_tokens(toks), row_key=row_label))
    return obs


def observations_for_cell(cell: Cell) -> List[Observation]:
    if cell.cell_type != "code":
        return []
    obs: List[Observation] = []
    last_code = _last_code_line(cell.source_code)
    prints = _print_lines(cell.source_code)
    bare_line_idx = 0
    for out in cell.outputs:
        if out.output_type == "error":
            continue
        if out.html_tables:
            for table in out.html_tables:
                obs.extend(_table_observations(cell, table))
            continue
        # Whitespace-aligned text tables (classification_report, DataFrame.to_string): a number-free
        # line of words is a header; following rows map their numbers onto it right-aligned.
        header_cols: Optional[List[str]] = None
        for line in out.text.splitlines():
            if not line.strip():
                header_cols = None
                continue
            mentions = extract_numbers(line)
            if not mentions:
                words = line.split()
                header_cols = words if len(words) >= 2 and all(_LETTERS_RE.search(w) for w in words) else None
                continue
            has_letters = bool(_LETTERS_RE.search(line))
            bare_label = ""
            if not has_letters:
                if out.output_type == "execute_result":
                    bare_label = last_code
                elif prints:
                    bare_label = prints[min(bare_line_idx, len(prints) - 1)]
                bare_line_idx += 1
            # Shape tuples like (420, 5) printed from df.shape
            shape = re.fullmatch(r"\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*", line)
            is_shape = bool(shape) and "shape" in (bare_label or line).lower()
            prev_label = ""
            in_table = bool(header_cols) and len(mentions) >= 2
            row_key = ""
            if in_table:
                row_label = line[: mentions[0].start].strip()
                first_col = len(header_cols) - len(mentions)
                if _LETTERS_RE.search(row_label):
                    row_key = row_label  # "LogReg  0.85  0.80" -> row "LogReg"
                elif 0 <= first_col < len(header_cols) and not kinds_of_text(header_cols[first_col]) - {"epochs"}:
                    row_key = f"{header_cols[first_col]} = {mentions[0].raw}"  # "max_depth = 8"
            for k, m in enumerate(mentions):
                kinds, quals = set(m.kinds), set(m.qualifiers)
                label = m.before.strip(" :=-|,\t")
                if in_table:
                    col_i = len(header_cols) - len(mentions) + k
                    col = header_cols[col_i] if 0 <= col_i < len(header_cols) else ""
                    row_kinds = kinds_of_text(row_label)
                    label = row_label if (row_kinds and k == 0) else f"{row_label} {col}".strip()
                    kinds = row_kinds if (row_kinds and k == 0) else kinds_of_text(label)
                    quals = qualifiers_of_tokens(tokenize(label))
                elif is_shape:
                    label = f"{bare_label} {'rows' if k == 0 else 'columns'}"
                    kinds = {"count"} if k == 0 else {"features"}
                elif not _LETTERS_RE.search(m.before) and k > 0 and prev_label:
                    label = prev_label  # "loss: [0.9, 0.8, 0.7]" -> inherit "loss"
                    kinds, quals = kinds_of_text(label), qualifiers_of_tokens(tokenize(label))
                elif not has_letters and bare_label:
                    label = bare_label
                    kinds = kinds_of_text(bare_label)
                    quals = qualifiers_of_tokens(tokenize(bare_label))
                if not kinds and label:
                    k2, q2 = nearest_kinds(label, "")
                    kinds, quals = k2 or kinds, quals | q2
                prev_label = label or prev_label
                obs.append(_mk_obs(cell, m, label, line, kinds=kinds, quals=quals, row_key=row_key))
    return obs


def code_observations(cell: Cell) -> List[Observation]:
    """Numeric literals bound to keyword args / names in code (e.g. epochs=10, test_size=0.2)."""
    obs: List[Observation] = []
    for line in cell.source_code.splitlines():
        for m in re.finditer(r"([A-Za-z_][\w.]*)\s*=\s*(-?\d+(?:\.\d+)?)\b", line):
            name, raw = m.group(1), m.group(2)
            mention = extract_numbers(raw)
            if not mention:
                continue
            label = name.replace("_", " ")
            obs.append(
                _mk_obs(cell, mention[0], label, line, location="code", kinds=kinds_of_text(label), quals=qualifiers_of_tokens(tokenize(label)))
            )
    return obs


def extract_observations(nb: Notebook, include_code: bool = True) -> List[Observation]:
    out: List[Observation] = []
    for cell in nb.code_cells:
        out.extend(observations_for_cell(cell))
        if include_code:
            out.extend(code_observations(cell))
    for i, o in enumerate(out):
        o.order = i
    return out


# --------------------------------------------------------------------------- number checks
def _item(o: Observation, relation: str) -> EvidenceItem:
    return EvidenceItem(cell_index=o.cell_index, location=o.location, snippet=o.line, value=o.value, label=o.label, relation=relation)


def _lexical_overlap(claim_text: str, o: Observation) -> int:
    return len(content_tokens(claim_text) & content_tokens(f"{o.label} {o.line}"))


def find_derivation(n: NumberMention, pool: Sequence[Observation]) -> Optional[Tuple[Observation, Observation, str]]:
    """Find a, b in pool such that the claimed number equals a-b, (a-b)/b or a/b."""
    uniq: Dict[float, Observation] = {}
    for o in pool:
        if o.location == "output":
            uniq.setdefault(o.value / 100 if o.is_percent else o.value, o)
    items = list(uniq.items())[:30]
    for (va, oa), (vb, ob) in permutations(items, 2):
        if va == vb:
            continue
        candidates = [(va - vb, "difference"), (va / vb if vb else None, "ratio")]
        if vb:
            candidates.append(((va - vb) / vb, "relative change"))
        for val, op in candidates:
            if val is None:
                continue
            decimals = 4 if abs(val) < 1 else 2
            fake = Observation(oa.cell_index, "output", val, f"{val:.6g}", False, decimals, op, [], [], "")
            if values_match(n, fake):
                return oa, ob, op
    return None


def check_numbers(claim: Claim, observations: Sequence[Observation]) -> List[NumberCheck]:
    checks: List[NumberCheck] = []
    outputs = [o for o in observations if o.location == "output"]
    for n in claim.numbers:
        direct = [o for o in observations if values_match(n, o)]
        comp_out = [
            o for o in outputs
            if comparable(n.kinds, n.qualifiers, o.kinds, o.qualifiers)
            # broad kinds ("count", "features") need a shared word: "252 positive reviews" vs "Loaded 420 documents" differ
            and (not set(n.kinds) <= BROAD_KINDS or _lexical_overlap(claim.text, o) >= 1)
        ]
        strong = [o for o in direct if comparable(n.kinds, n.qualifiers, o.kinds, o.qualifiers)]
        lexical = [o for o in direct if not (n.kinds and o.kinds) and _lexical_overlap(claim.text, o) >= 1]
        if strong or lexical:
            hits = sorted(strong or lexical, key=lambda o: (o.location != "output", o.order))[:3]
            note = "label and value match" if strong else "value matches and label shares terms with the claim"
            checks.append(NumberCheck(n.raw, "match", [_item(o, "match") for o in hits], note))
            continue
        # Only attempt derivations among same-kind outputs; free search over all numbers is spurious.
        derivation = find_derivation(n, comp_out) if comp_out else None
        if derivation:
            a, b, op = derivation
            checks.append(
                NumberCheck(
                    n.raw,
                    "derived",
                    [_item(a, "derived"), _item(b, "derived")],
                    f"{n.raw} = {op} of {fmt_value(a.value)} (Cell {a.cell_index}) and {fmt_value(b.value)} (Cell {b.cell_index})",
                )
            )
            continue
        if comp_out and n.qualifier != "range":
            closest = sorted(comp_out, key=lambda o: abs((o.value / 100 if o.is_percent else o.value) - (n.value / 100 if n.is_percent else n.value)))[:2]
            note = f"claim states {n.raw}; notebook prints " + ", ".join(f"{o.raw} ({o.label or 'unlabelled'}, Cell {o.cell_index})" for o in closest)
            checks.append(NumberCheck(n.raw, "mismatch", [_item(o, "mismatch") for o in closest], note))
            continue
        if direct:
            hits = sorted(direct, key=lambda o: o.order)[:2]
            checks.append(
                NumberCheck(n.raw, "weak_match", [_item(o, "weak_match") for o in hits], "value appears but under an unrelated label")
            )
            continue
        checks.append(NumberCheck(n.raw, "untraced", [], "no printed output contains this value"))
    return checks


def output_series(observations: Sequence[Observation], min_points: int = 3) -> List[SeriesEvidence]:
    """Group labelled output numbers (e.g. per-epoch logs) into series."""
    groups: Dict[Tuple[int, str, Tuple[str, ...]], List[Observation]] = {}
    for o in observations:
        if o.location != "output" or not o.kinds or o.kinds == ["epochs"]:
            continue
        key = (o.cell_index, o.kinds[0], tuple(o.qualifiers))
        groups.setdefault(key, []).append(o)
    series: List[SeriesEvidence] = []
    for (cell, kind, quals), obs in groups.items():
        if len(obs) < min_points:
            continue
        vals = [o.value for o in sorted(obs, key=lambda o: o.order)]
        d, mono = series_direction(vals)
        label = " ".join(list(quals) + [kind])
        series.append(SeriesEvidence(cell, label, vals, d, mono, "output"))
    return series


# --------------------------------------------------------------------------- auditor
class EvidenceAuditor:
    """Builds the claim -> evidence map and emits discrepancy findings."""

    engine = "evidence"

    def build_map(self, claims: Sequence[Claim], verdicts: Sequence[ClaimVerdict], executions: Sequence[ExecutionEvidence]) -> List[dict]:
        ex_by = {e.claim_id: e for e in executions}
        v_by = {v.claim_id: v for v in verdicts}
        rows = []
        for c in claims:
            v = v_by.get(c.claim_id)
            ex = ex_by.get(c.claim_id)
            ev = v.evidence if v else []
            rows.append(
                {
                    "claim_id": c.claim_id,
                    "claim": c.text,
                    "source": c.source,
                    "verdict": v.verdict if v else None,
                    "cells": sorted({e.cell_index for e in ev}),
                    "evidence": [
                        {"cell": e.cell_index, "location": e.location, "relation": e.relation, "snippet": e.snippet, "label": e.label}
                        for e in ev
                    ],
                    "number_checks": [{"number": nc.raw, "status": nc.status, "note": nc.note} for nc in (ex.number_checks if ex else [])],
                }
            )
        return rows

    def findings(self, claims: Sequence[Claim], executions: Sequence[ExecutionEvidence]) -> List[RiskFinding]:
        out: List[RiskFinding] = []
        by_id = {c.claim_id: c for c in claims}
        for ex in executions:
            claim = by_id.get(ex.claim_id)
            if claim is None:
                continue
            for nc in ex.number_checks:
                cells = sorted({e.cell_index for e in nc.evidence})
                if nc.status == "mismatch":
                    out.append(
                        RiskFinding(
                            "", self.engine, "numeric_discrepancy", "high",
                            f"{claim.claim_id}: claim says {nc.raw} but {_cells(cells)} print{'s' if len(cells) == 1 else ''} "
                            + ", ".join(f"`{e.snippet}`" for e in nc.evidence[:2]),
                            cells, nc.note, [claim.claim_id],
                        )
                    )
                elif nc.status == "weak_match":
                    out.append(
                        RiskFinding(
                            "", self.engine, "ambiguous_evidence", "low",
                            f"{claim.claim_id}: {nc.raw} appears in {_cells(cells)} but under an unrelated label",
                            cells, nc.note, [claim.claim_id],
                        )
                    )
            untraced = [nc.raw for nc in ex.number_checks if nc.status == "untraced"]
            if untraced:
                metric = any(set(n.kinds) - {"count", "features", "classes", "epochs"} for n in claim.numbers)
                out.append(
                    RiskFinding(
                        "", self.engine, "untraceable_number", "medium" if metric else "low",
                        f"{claim.claim_id}: {', '.join(untraced)} not found in any cell output",
                        [], f"Claim: {claim.text}", [claim.claim_id],
                    )
                )
        return out


def _cells(cells: Sequence[int]) -> str:
    if not cells:
        return "no cell"
    return ("Cell " if len(cells) == 1 else "Cells ") + ", ".join(str(c) for c in cells)
