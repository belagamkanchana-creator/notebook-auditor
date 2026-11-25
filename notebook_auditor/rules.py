"""Strict, deterministic verification rules used by the Verifier agent.

The LLM verifier may only *upgrade* an UNVERIFIABLE result, and only with verbatim citations;
these rules are authoritative for exact numbers, trends with data, and code-level procedures.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from .code_analysis import CodeIndex
from .models import Claim, EvidenceItem, NumberCheck, RiskFinding, SeriesEvidence
from .textnum import EVAL_KINDS, comparable, content_tokens, kinds_of_text, qualifiers_of_tokens, tokenize

SEVERITY_ORDER = {"CONTRADICTED": 3, "METHODOLOGICAL_RISK": 2, "UNVERIFIABLE": 1, "SUPPORTED": 0}
LEAK_CHECKS = {"data_leakage", "test_set_contamination", "group_leakage", "no_holdout"}
BASELINE_EVIDENCE_RE = re.compile(r"DummyClassifier|DummyRegressor|most_frequent|majority|baseline|chance", re.I)
DECISION_RE = re.compile(r"significan|exceed|meets?|pass(es|ed)?|above|surpass|beat|threshold|target|cut-?off|better than", re.I)


class RuleResult:
    def __init__(self, verdict: str, confidence: float, rationale: str, rules: List[str], evidence: List[EvidenceItem]) -> None:
        self.verdict = verdict
        self.confidence = confidence
        self.rationale = rationale
        self.rules = rules
        self.evidence = evidence

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "confidence": self.confidence, "rationale": self.rationale, "rules": self.rules}


def worst(results: Sequence[Optional[RuleResult]]) -> Optional[RuleResult]:
    real = [r for r in results if r is not None]
    if not real:
        return None
    top = max(real, key=lambda r: SEVERITY_ORDER[r.verdict])
    merged_rules = [rule for r in real for rule in r.rules]
    merged_ev = [e for r in real for e in r.evidence]
    rationale = " ".join(r.rationale for r in sorted(real, key=lambda r: -SEVERITY_ORDER[r.verdict]))
    return RuleResult(top.verdict, top.confidence, rationale, merged_rules, merged_ev)


# --------------------------------------------------------------------------- numeric
def verify_numbers(checks: Sequence[NumberCheck]) -> Optional[RuleResult]:
    if not checks:
        return None
    ev = [e for c in checks for e in c.evidence]
    statuses = [c.status for c in checks]
    mism = [c for c in checks if c.status == "mismatch"]
    if mism:
        return RuleResult("CONTRADICTED", 0.85, " ".join(f"N-MISMATCH: {c.note}." for c in mism), ["N-MISMATCH"], ev)
    if all(s in ("match", "derived") for s in statuses):
        derived = [c for c in checks if c.status == "derived"]
        if derived:
            return RuleResult("SUPPORTED", 0.75, " ".join(f"N-DERIVED: {c.note}." for c in derived) + " Remaining numbers printed directly.", ["N-MATCH", "N-DERIVED"], ev)
        return RuleResult("SUPPORTED", 0.9, "N-MATCH: every number in the claim is printed by a cell with a matching label.", ["N-MATCH"], ev)
    ok = sum(1 for s in statuses if s in ("match", "derived"))
    weak = [c for c in checks if c.status == "weak_match"]
    msg = f"N-PARTIAL: {ok} of {len(checks)} numbers traced to outputs."
    if weak:
        msg += " " + " ".join(f"{c.raw}: {c.note}." for c in weak)
    untraced = [c.raw for c in checks if c.status == "untraced"]
    if untraced:
        msg += f" Not found in any output: {', '.join(untraced)}."
    return RuleResult("UNVERIFIABLE", 0.6, msg, ["N-PARTIAL" if ok else "N-UNTRACED"], ev)


# --------------------------------------------------------------------------- trend
def _claim_kinds_quals(claim: Claim) -> Tuple[set, set]:
    kinds = {k for n in claim.numbers for k in n.kinds} | kinds_of_text(claim.text)
    kinds.discard("epochs")
    return kinds, qualifiers_of_tokens(tokenize(claim.text))


def verify_trend(claim: Claim, series: Sequence[SeriesEvidence]) -> Optional[RuleResult]:
    if not claim.direction or "comparison" in claim.tags:
        return None  # "A is higher than B" is a comparison, not a trend over a series
    # Internal consistency: "increased from 0.71 to 0.85"
    m = re.search(r"\bfrom\b(.+?)\bto\b", claim.text, re.I)
    if m and len(claim.numbers) >= 2:
        a, b = claim.numbers[0].value, claim.numbers[1].value
        actual = "increase" if b > a else "decrease" if b < a else "stable"
        if actual != claim.direction and claim.direction in ("increase", "decrease"):
            return RuleResult("CONTRADICTED", 0.9, f"T-INTERNAL: claim says '{claim.direction}' but its own numbers go {a:g} -> {b:g}.", ["T-INTERNAL"], [])
    kinds, quals = _claim_kinds_quals(claim)
    ctoks = content_tokens(claim.text)
    scored = []
    for s in series:
        if not s.direction or s.direction == "mixed" and "monotonic" not in claim.tags:
            continue
        sk = kinds_of_text(s.label)
        sq = qualifiers_of_tokens(tokenize(s.label))
        score = 3 if comparable(sorted(kinds), sorted(quals), sorted(sk), sorted(sq)) else 2 if kinds & sk else 1 if len(ctoks & content_tokens(s.label)) >= 1 else 0
        if score:
            scored.append((score, s))
    if not scored:
        return RuleResult("UNVERIFIABLE", 0.5, "T-NODATA: no printed or plotted series for this quantity was found.", ["T-NODATA"], []) if claim.category == "trend" else None
    best = max(sc for sc, _ in scored)
    top = [s for sc, s in scored if sc == best]
    if best < 2:  # only shares a word with the claim, not the metric: too weak to support or contradict
        s = top[0]
        return RuleResult("UNVERIFIABLE", 0.4, f"T-WEAK: the closest series ({s.label}, Cell {s.cell_index}) does not measure the claimed quantity.", ["T-WEAK"],
                          [EvidenceItem(s.cell_index, "output" if s.source == "output" else "code", f"{s.label}: {_fmt_series(s.values)}", None, s.label, "context")]) if claim.category == "trend" else None
    ev = [EvidenceItem(s.cell_index, "output" if s.source == "output" else "code", f"{s.label}: {_fmt_series(s.values)}", None, s.label, "") for s in top]
    for s in top:
        if s.direction != claim.direction:
            for e in ev:
                e.relation = "mismatch"
            return RuleResult("CONTRADICTED", 0.8 if best >= 2 else 0.6,
                              f"T-DIRECTION: claim says '{claim.direction}' but {s.label} in Cell {s.cell_index} shows '{s.direction}' ({_fmt_series(s.values)}).",
                              ["T-DIRECTION"], ev)
        if "monotonic" in claim.tags and not s.monotonic:
            for e in ev:
                e.relation = "mismatch"
            return RuleResult("CONTRADICTED", 0.7,
                              f"T-MONOTONIC: claim says the {claim.direction} is steady/consistent, but {s.label} in Cell {s.cell_index} reverses direction ({_fmt_series(s.values)}).",
                              ["T-MONOTONIC"], ev)
    for e in ev:
        e.relation = "match"
    return RuleResult("SUPPORTED", 0.8 if best >= 2 else 0.6, f"T-DIRECTION: {top[0].label} in Cell {top[0].cell_index} shows a {claim.direction} ({_fmt_series(top[0].values)}).", ["T-DIRECTION"], ev)


def _fmt_series(values: Sequence[float]) -> str:
    vals = [f"{v:.4g}" for v in values]
    return "[" + ", ".join(vals if len(vals) <= 8 else vals[:4] + ["…"] + vals[-3:]) + "]"


# --------------------------------------------------------------------------- methodological
def _code_item(idx: CodeIndex, cell: int, line: str, relation: str) -> EvidenceItem:
    return EvidenceItem(cell, "code", line, None, "", relation)


def _split_ratio(text: str) -> Optional[float]:
    t = text.lower()
    m = re.search(r"\b(\d{1,2})\s*[/:-]\s*(\d{1,2})\b", t)
    if m and int(m.group(1)) + int(m.group(2)) == 100:
        return min(int(m.group(1)), int(m.group(2))) / 100
    m = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%\s*(?:of the data\s*)?(?:was |were |is )?(?:held out |used |reserved |set aside )?(?:for |as )?(?:the |a )?(?:test|validation|held-?out|hold-?out)", t)
    if m:
        return float(m.group(1)) / 100
    m = re.search(r"test[_ ]size\s*(?:of|=|:)?\s*(0?\.\d+)", t)
    if m:
        return float(m.group(1))
    return None


def verify_method(claim: Claim, idx: CodeIndex, risks: Sequence[RiskFinding]) -> Optional[RuleResult]:
    t = claim.text.lower()
    checks: List[Tuple[str, str, str, List[EvidenceItem]]] = []  # rule, status, message, evidence
    splits = idx.calls_to("train_test_split")
    kfolds = [c for c in idx.calls if c.attr in ("KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "RepeatedKFold", "RepeatedStratifiedKFold")]
    cvs = [c for c in idx.calls if c.attr in ("cross_val_score", "cross_validate", "cross_val_predict", "GridSearchCV", "RandomizedSearchCV")]
    leak = [r for r in risks if r.check in LEAK_CHECKS and r.severity == "high"]

    if "stratif" in t:
        strat = [c for c in splits if c.kwargs.get("stratify", "None") != "None"] + [c for c in idx.calls if c.attr.startswith("Stratified")]
        if strat:
            checks.append(("M-STRATIFY", "support", "split is stratified", [_code_item(idx, c.cell_index, c.source_line, "match") for c in strat[:2]]))
        elif splits or kfolds:
            src = (splits or kfolds)[0]
            checks.append(("M-STRATIFY", "contradict", f"claim says stratified, but `{src.source_line}` (Cell {src.cell_index}) has no stratify argument", [_code_item(idx, src.cell_index, src.source_line, "mismatch")]))
        else:
            checks.append(("M-STRATIFY", "unknown", "no split code found", []))

    ratio = _split_ratio(t) if re.search(r"split|test|held|hold|validation", t) else None
    if ratio is not None:
        if splits:
            c = splits[0]
            if "test_size" in c.kwargs:
                actual = c.kwarg_values.get("test_size")
                if actual is not None and actual >= 1:
                    actual = None  # absolute row counts cannot be compared to a ratio
            elif "train_size" in c.kwargs and c.kwarg_values.get("train_size") is not None:
                actual = 1 - c.kwarg_values["train_size"]  # type: ignore[operator]
            else:
                actual = 0.25  # scikit-learn default
            if actual is None:
                checks.append(("M-SPLIT-RATIO", "unknown", f"test_size in Cell {c.cell_index} is not a literal fraction", []))
            elif abs(actual - ratio) <= 0.011:
                checks.append(("M-SPLIT-RATIO", "support", f"test fraction {actual:g} matches {ratio:g}", [_code_item(idx, c.cell_index, c.source_line, "match")]))
            else:
                checks.append(("M-SPLIT-RATIO", "contradict", f"claim implies a test fraction of {ratio:g} but `{c.source_line}` (Cell {c.cell_index}) gives {actual:g}" + (" (sklearn default)" if "test_size" not in c.kwargs and "train_size" not in c.kwargs else ""), [_code_item(idx, c.cell_index, c.source_line, "mismatch")]))
        else:
            checks.append(("M-SPLIT-RATIO", "unknown", "no train_test_split call found", []))

    m = re.search(r"(\d+)[- ]fold", t)
    if m or re.search(r"cross[- ]?valid", t):
        if kfolds or cvs:
            if m:
                want = int(m.group(1))
                found = []
                for c in kfolds + cvs:
                    v = c.kwarg_values.get("n_splits") or c.kwarg_values.get("cv")
                    if v is None and c in kfolds and c.arg_nodes:
                        from .code_analysis import literal_number

                        v = literal_number(c.arg_nodes[0])
                    if v is None and ("cv" not in c.kwargs and "n_splits" not in c.kwargs):
                        v = 5.0  # sklearn default
                    if v is not None:
                        found.append((v, c))
                if any(int(v) == want for v, _ in found):
                    c = next(c for v, c in found if int(v) == want)
                    checks.append(("M-CV", "support", f"{want}-fold CV in Cell {c.cell_index}", [_code_item(idx, c.cell_index, c.source_line, "match")]))
                elif found:
                    v, c = found[0]
                    checks.append(("M-CV", "contradict", f"claim says {want}-fold but `{c.source_line}` (Cell {c.cell_index}) uses {int(v)} folds", [_code_item(idx, c.cell_index, c.source_line, "mismatch")]))
                else:
                    checks.append(("M-CV", "unknown", "fold count not a literal", []))
            else:
                c = (kfolds + cvs)[0]
                checks.append(("M-CV", "support", f"cross-validation in Cell {c.cell_index}", [_code_item(idx, c.cell_index, c.source_line, "match")]))
        elif splits:
            c = splits[0]
            checks.append(("M-CV", "contradict", f"claim says cross-validation but only a single train_test_split (Cell {c.cell_index}) is used", [_code_item(idx, c.cell_index, c.source_line, "mismatch")]))
        else:
            checks.append(("M-CV", "unknown", "no cross-validation code found", []))

    if re.search(r"random seed|random state|seed|reproducib", t):
        hits = idx.source_matches(r"random_state\s*=\s*(?!None)|np\.random\.seed|random\.seed|torch\.manual_seed|set_seed|seed_everything")
        if hits:
            checks.append(("M-SEED", "support", f"seed fixed in Cell {hits[0][0]}", [_code_item(idx, hits[0][0], hits[0][2], "match")]))
        else:
            checks.append(("M-SEED", "contradict", "claim says seeds were fixed but no random_state/seed call exists", []))

    if re.search(r"baseline|majority[- ]class|chance|dummy", t):
        hits = idx.source_matches(BASELINE_EVIDENCE_RE.pattern)
        if hits:
            checks.append(("M-BASELINE", "support", f"baseline computed in Cell {hits[0][0]}", [_code_item(idx, hits[0][0], hits[0][2], "match")]))
        else:
            checks.append(("M-BASELINE", "unknown", "claim refers to a baseline/chance level, but no baseline is computed in the notebook", []))

    if re.search(r"standardi[sz]|normali[sz]|\bscal(ed|ing|er)\b", t):
        hits = idx.source_matches(r"Scaler\(|Normalizer\(|normalize\(|\.std\(\)")
        if hits:
            checks.append(("M-SCALING", "support", f"scaling in Cell {hits[0][0]}", [_code_item(idx, hits[0][0], hits[0][2], "match")]))
        else:
            checks.append(("M-SCALING", "unknown", "no scaling code found", []))
        if re.search(r"after (the )?split|only on (the )?train|fit(ted)? on (the )?train|no leakage|without leakage", t) and leak:
            checks.append(("M-LEAKAGE", "contradict", f"claim says transforms were fitted on training data only, but {leak[0].message}", []))

    if re.search(r"held[- ]?out|hold[- ]?out|unseen|never seen|test set", t):
        if splits or kfolds or cvs:
            c = (splits or kfolds or cvs)[0]
            checks.append(("M-HOLDOUT", "support", f"held-out split in Cell {c.cell_index}", [_code_item(idx, c.cell_index, c.source_line, "match")]))
        else:
            checks.append(("M-HOLDOUT", "contradict", "claim mentions a held-out/test set but the notebook never splits the data", []))

    if re.search(r"duplicat", t):
        hits = idx.source_matches(r"drop_duplicates|duplicated\(")
        checks.append(("M-DEDUP", "support" if hits else "unknown", f"deduplication in Cell {hits[0][0]}" if hits else "no deduplication code found", [_code_item(idx, hits[0][0], hits[0][2], "match")] if hits else []))

    if re.search(r"imput|missing values", t):
        hits = idx.source_matches(r"fillna|Imputer|dropna|interpolate\(")
        checks.append(("M-IMPUTE", "support" if hits else "unknown", f"missing-value handling in Cell {hits[0][0]}" if hits else "no missing-value handling found", [_code_item(idx, hits[0][0], hits[0][2], "match")] if hits else []))

    if re.search(r"tf-?idf|vectori[sz]|bag[- ]of[- ]words", t):
        hits = idx.source_matches(r"Vectorizer\(")
        checks.append(("M-VECTORIZE", "support" if hits else "unknown", f"vectoriser in Cell {hits[0][0]}" if hits else "no vectoriser found", [_code_item(idx, hits[0][0], hits[0][2], "match")] if hits else []))

    if not checks:
        return None
    ev = [e for _, _, _, items in checks for e in items]
    rules = [r for r, _, _, _ in checks]
    contra = [msg for _, st, msg, _ in checks if st == "contradict"]
    unknown = [msg for _, st, msg, _ in checks if st == "unknown"]
    support = [msg for _, st, msg, _ in checks if st == "support"]
    if contra:
        return RuleResult("CONTRADICTED", 0.85, "M-CODE: " + "; ".join(contra) + ".", rules, ev)
    if unknown:
        prefix = f"M-PARTIAL: verified {'; '.join(support)}; " if support else "M-NOCODE: "
        return RuleResult("UNVERIFIABLE", 0.6, prefix + "; ".join(unknown) + ".", rules, ev)
    return RuleResult("SUPPORTED", 0.85, "M-CODE: " + "; ".join(support) + ".", rules, ev)


# --------------------------------------------------------------------------- risk overlay
def related_risks(claim: Claim, risks: Sequence[RiskFinding], plan_cells: Sequence[int]) -> List[RiskFinding]:
    kinds = {k for n in claim.numbers for k in n.kinds} | kinds_of_text(claim.text)
    is_eval = bool(kinds & EVAL_KINDS) or bool(re.search(r"perform|generali|predict|classif", claim.text, re.I))
    is_split_claim = bool(re.search(r"split|held|test set|scal|standardi|normali|leak|unseen", claim.text, re.I))
    out = []
    for r in risks:
        if claim.claim_id in r.claim_ids:
            out.append(r)
        elif r.check in LEAK_CHECKS and r.severity in ("high", "medium") and (is_eval or is_split_claim):
            out.append(r)
        elif r.check in ("missing_baseline", "near_chance") and is_eval and kinds & {"accuracy", "f1", "precision", "recall", "auc"}:
            out.append(r)
        elif r.check == "borderline_threshold" and set(r.cell_indices) & set(plan_cells) and (kinds & set(tokenize(r.message)) or "p_value" in kinds):
            out.append(r)
    return out


def apply_risk_overlay(claim: Claim, result: RuleResult, risks: Sequence[RiskFinding]) -> RuleResult:
    if result.verdict != "SUPPORTED":
        return result
    blocking = []
    for r in risks:
        if r.check in LEAK_CHECKS and r.severity == "high":
            blocking.append(r)
        elif r.check == "near_chance":
            blocking.append(r)
        elif r.check == "borderline_threshold" and DECISION_RE.search(claim.text):
            blocking.append(r)
    if not blocking:
        return result
    msg = " ".join(f"RISK-{r.finding_id}: {r.message}." for r in blocking)
    return RuleResult("METHODOLOGICAL_RISK", min(result.confidence, 0.8),
                      result.rationale + f" The reported value matches the notebook, but {msg}", result.rules + ["RISK-OVERLAY"], result.evidence)
