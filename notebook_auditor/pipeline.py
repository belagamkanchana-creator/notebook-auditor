"""Planner -> Executor -> Verifier multi-agent audit pipeline.

Each agent has a deterministic core and an optional LLM layer. LLM output is never trusted
blindly: cell indices are range-checked, extracted values must be verbatim quotes from the
cell, and verdicts must carry verifiable citations.
"""
from __future__ import annotations

import datetime as _dt
import re
from math import sqrt
from typing import Dict, List, Optional, Sequence

from .auditors.claim_strength_auditor import ClaimStrengthAuditor
from .auditors.evidence_auditor import EvidenceAuditor, check_numbers, extract_observations, output_series
from .auditors.figure_auditor import FIGURE_REF_RE, FigureAuditor
from .auditors.risk_auditor import RiskAuditor
from .claim_extractor import extract_claims
from .code_analysis import CodeIndex
from .findings import FindingsBuilder, build_gaps
from .llm import LLMClient, make_client
from .models import (
    AuditResult,
    Claim,
    ClaimVerdict,
    EvidenceItem,
    ExecutionEvidence,
    Notebook,
    Observation,
    RiskFinding,
    TracePlan,
    VERDICTS,
    render_cell,
)
from .parser import load_notebook, notebook_summary
from .prompts import (
    EXECUTOR_SYSTEM,
    PLANNER_SYSTEM,
    VERIFIER_SYSTEM,
    executor_user_prompt,
    planner_user_prompt,
    verifier_user_prompt,
)
from .rules import RuleResult, apply_risk_overlay, related_risks, verify_method, verify_numbers, verify_trend, worst
from .textnum import comparable, content_tokens, extract_numbers, kinds_of_text, qualifiers_of_tokens, tokenize, values_match

METHOD_CODE_HINTS = [
    (r"split|held|hold|test set|train/test|stratif|\d+\s*/\s*\d+", r"train_test_split|KFold|ShuffleSplit|stratify"),
    (r"scal|standardi|normali", r"Scaler|normalize|Normalizer"),
    (r"cross[- ]?valid|fold", r"cross_val|KFold|cv\s*="),
    (r"baseline|majority|chance|dummy", r"Dummy|majority|baseline|most_frequent|chance"),
    (r"seed|reproduc", r"random_state|seed"),
    (r"duplicat", r"drop_duplicates|duplicated"),
    (r"imput|missing", r"fillna|Imputer|dropna|isna|isnull"),
    (r"tf-?idf|vectori", r"Vectorizer"),
    (r"smote|oversampl|undersampl", r"SMOTE|Sampler"),
    (r"leak", r"fit_transform|fit\(|train_test_split"),
]


def _strip_imports(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if not re.match(r"\s*(import|from)\s+\S+", l) and not l.lstrip().startswith(("%", "!")))


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _claim_payload(c: Claim) -> dict:
    return {"claim_id": c.claim_id, "claim": c.text, "category": c.category, "context": c.context}


# --------------------------------------------------------------------------- Planner
class PlannerAgent:
    """Maps each claim to the cells (code + outputs) that should contain its evidence."""

    def __init__(self, nb: Notebook, observations: Sequence[Observation], figures: FigureAuditor, llm: Optional[LLMClient] = None, max_cells: int = 4) -> None:
        self.nb = nb
        self.llm = llm
        self.max_cells = max_cells
        self.figures = figures
        self.obs_by_cell: Dict[int, List[Observation]] = {}
        for o in observations:
            self.obs_by_cell.setdefault(o.cell_index, []).append(o)
        # Import lines mention every metric/splitter name, so they are not evidence of anything.
        self.code_wo_imports = {c.cell_index: _strip_imports(c.source_code) for c in nb.code_cells}
        self.cell_tokens = {c.cell_index: content_tokens(f"{self.code_wo_imports[c.cell_index]}\n{c.output_text}") for c in nb.code_cells}
        self.warnings: List[str] = []

    def _score(self, claim: Claim, cell_index: int) -> (float, List[str]):
        score, reasons = 0.0, []
        obs = self.obs_by_cell.get(cell_index, [])
        for n in claim.numbers:
            hit = next((o for o in obs if values_match(n, o) and comparable(n.kinds, n.qualifiers, o.kinds, o.qualifiers)), None)
            if hit:
                score += 4
                reasons.append(f"prints {hit.raw} ({hit.label or 'value'})")
                continue
            if any(values_match(n, o) for o in obs):
                score += 1
            same_kind = next((o for o in obs if o.location == "output" and comparable(n.kinds, n.qualifiers, o.kinds, o.qualifiers)), None)
            if same_kind:
                score += 2.5
                reasons.append(f"prints {'/'.join(n.kinds)} ({same_kind.raw})")
        overlap = content_tokens(claim.text) & self.cell_tokens.get(cell_index, set())
        if overlap:
            score += min(len(overlap), 6) * 0.5
            reasons.append("shares terms: " + ", ".join(sorted(overlap)[:4]))
        src = self.code_wo_imports.get(cell_index, "")
        if claim.category == "methodological" or "methodological" in claim.tags:
            for claim_pat, code_pat in METHOD_CODE_HINTS:
                if re.search(claim_pat, claim.text, re.I) and re.search(code_pat, src):
                    score += 3
                    reasons.append(f"implements {code_pat.split('|')[0]}")
        if (claim.direction or FIGURE_REF_RE.search(claim.text)) and cell_index in self.figures.plot_cells:
            score += 1.5
            if any(s.cell_index == cell_index for s in self.figures.matching_series(claim)):
                score += 3
                reasons.append("plots the claimed series")
        return score, reasons

    def heuristic_plan(self, claim: Claim) -> TracePlan:
        scored = []
        for cell in self.nb.code_cells:
            s, why = self._score(claim, cell.cell_index)
            if s >= 1.5:
                scored.append((s, cell.cell_index, why))
        scored.sort(key=lambda t: (-t[0], t[1]))
        top = scored[: self.max_cells]
        rationale = "; ".join(f"Cell {i}: {', '.join(w) or 'related code'}" for _, i, w in top) or "no cell output or code relates to this claim"
        return TracePlan(claim.claim_id, sorted(i for _, i, _ in top), rationale, "heuristic")

    def _notebook_index(self) -> str:
        rows = []
        for c in self.nb.code_cells:
            code = _norm_ws(c.source_code)[:240]
            out = _norm_ws(c.output_text)[:240]
            rows.append(f"[Cell {c.cell_index}] code: {code}" + (f"\n   output: {out}" if out else ""))
        return "\n".join(rows)

    def plan(self, claims: Sequence[Claim]) -> List[TracePlan]:
        plans = {c.claim_id: self.heuristic_plan(c) for c in claims}
        if self.llm is None or not claims:
            return [plans[c.claim_id] for c in claims]
        valid = {c.cell_index for c in self.nb.code_cells}
        index = self._notebook_index()
        for start in range(0, len(claims), 25):
            batch = claims[start : start + 25]
            data = self.llm.complete_json(PLANNER_SYSTEM, planner_user_prompt(index, [_claim_payload(c) for c in batch]), max_tokens=3000)
            if not isinstance(data, dict) or not isinstance(data.get("plans"), list):
                self.warnings.append("LLM planner failed; using heuristic plans for a batch.")
                continue
            for item in data["plans"]:
                cid = str(item.get("claim_id", ""))
                if cid not in plans:
                    continue
                cells = []
                for x in item.get("target_cells") or []:
                    try:
                        xi = int(x)
                    except (TypeError, ValueError):
                        continue
                    if xi in valid and xi not in cells:
                        cells.append(xi)
                heur = plans[cid]
                merged = cells + [i for i in heur.target_cells if i not in cells]
                plans[cid] = TracePlan(cid, sorted(merged[: self.max_cells + 2]), f"LLM: {item.get('rationale', '')} | heuristic: {heur.rationale}", "llm+heuristic")
        return [plans[c.claim_id] for c in claims]


# --------------------------------------------------------------------------- Executor
class ExecutorAgent:
    """Pulls exact values (numbers, dims, code parameters, series) out of the planned cells."""

    def __init__(self, nb: Notebook, observations: Sequence[Observation], figures: FigureAuditor, llm: Optional[LLMClient] = None) -> None:
        self.nb = nb
        self.obs = list(observations)
        self.figures = figures
        self.llm = llm

    def _llm_observations(self, claim: Claim, cells: Sequence[int]) -> (List[Observation], List[EvidenceItem], List[str]):
        cells_text = "\n\n".join(self.nb.cell_text(i, max_output_chars=3000) for i in cells)
        data = self.llm.complete_json(EXECUTOR_SYSTEM, executor_user_prompt(_claim_payload(claim), cells_text), max_tokens=2000) if self.llm else None
        if not isinstance(data, dict) or not isinstance(data.get("values"), list):
            return [], [], ["LLM executor returned no usable values"]
        obs, facts, notes = [], [], []
        for item in data["values"]:
            try:
                ci = int(item.get("cell_index"))
            except (TypeError, ValueError):
                continue
            cell = self.nb.get(ci)
            quote = str(item.get("quote") or "")
            if cell is None or ci not in cells or not quote.strip():
                continue
            haystack = _norm_ws(cell.source_code + "\n" + cell.output_text)
            if _norm_ws(quote) not in haystack:
                notes.append(f"Rejected LLM value not found verbatim in Cell {ci}: {quote[:60]!r}")
                continue
            label = str(item.get("label") or "")
            loc = "code" if _norm_ws(quote) in _norm_ws(cell.source_code) and _norm_ws(quote) not in _norm_ws(cell.output_text) else "output"
            nums = extract_numbers(quote)
            target = item.get("value")
            chosen = None
            if nums:
                chosen = nums[0]
                if isinstance(target, (int, float)):
                    chosen = min(nums, key=lambda m: abs(m.value - float(target)))
            if chosen is None:
                facts.append(EvidenceItem(ci, loc, quote.strip()[:200], None, label, "context"))
                continue
            toks = tokenize(label)
            obs.append(
                Observation(ci, loc, chosen.value, chosen.raw, chosen.is_percent, chosen.decimals, label,
                            sorted(kinds_of_text(label) or set(chosen.kinds)), sorted(qualifiers_of_tokens(toks) | set(chosen.qualifiers)),
                            quote.strip()[:200], 10**6 + len(obs), "llm")
            )
        return obs, facts, notes

    def execute(self, claim: Claim, plan: TracePlan) -> ExecutionEvidence:
        cells = set(plan.target_cells)
        pool = [o for o in self.obs if o.cell_index in cells]
        ex = ExecutionEvidence(claim.claim_id)
        # Numbers in procedural claims ("80/20 split", "5-fold") are checked against code by the method rules.
        ex.number_checks = check_numbers(claim, pool) if claim.category != "methodological" else []
        ex.series = output_series(pool) + [s for s in self.figures.series if s.cell_index in cells]
        if claim.direction:
            for s in self.figures.matching_series(claim):
                if s not in ex.series:
                    ex.series.append(s)
        for i in sorted(cells):
            cell = self.nb.get(i)
            if cell is None:
                continue
            for line in cell.source_code.splitlines():
                if any(re.search(cp, claim.text, re.I) and re.search(kp, line) for cp, kp in METHOD_CODE_HINTS):
                    ex.code_facts.append(EvidenceItem(i, "code", line.strip(), None, "", "context"))
        needs_llm = self.llm is not None and cells and (
            any(nc.status not in ("match", "derived") for nc in ex.number_checks) or claim.category in ("trend", "qualitative", "methodological")
        )
        if needs_llm:
            llm_obs, facts, notes = self._llm_observations(claim, sorted(cells))
            ex.notes.extend(notes)
            ex.code_facts.extend(facts)
            if llm_obs and claim.category != "methodological":
                ex.number_checks = check_numbers(claim, pool + llm_obs)
                ex.executor = "heuristic+llm"
        return ex


# --------------------------------------------------------------------------- Verifier
class VerifierAgent:
    """Applies strict rules; an LLM may only resolve UNVERIFIABLE cases, with verbatim citations."""

    def __init__(self, nb: Notebook, index: CodeIndex, risks: Sequence[RiskFinding], llm: Optional[LLMClient] = None) -> None:
        self.nb = nb
        self.idx = index
        self.risks = list(risks)
        self.llm = llm

    def verify(self, claim: Claim, plan: TracePlan, ex: ExecutionEvidence) -> ClaimVerdict:
        numeric = verify_numbers(ex.number_checks)
        trend = verify_trend(claim, ex.series)
        method = verify_method(claim, self.idx, self.risks) if (claim.category == "methodological" or "methodological" in claim.tags) else None
        result = worst([numeric, trend, method])
        if result is None:
            if claim.category == "qualitative" or "interpretive" in claim.tags:
                result = RuleResult("UNVERIFIABLE", 0.5, "Q-INTERPRETIVE: interpretive claim; printed outputs cannot establish it on their own.", ["Q-INTERPRETIVE"], [])
            else:
                result = RuleResult("UNVERIFIABLE", 0.4, "NO-EVIDENCE: no rule could connect this claim to a cell output or code.", ["NO-EVIDENCE"], [])
        related = related_risks(claim, self.risks, plan.target_cells)
        result = apply_risk_overlay(claim, result, related)
        verifier = "rules"
        if result.verdict == "UNVERIFIABLE" and self.llm is not None and plan.target_cells:
            llm_result = self._llm_verify(claim, plan, ex, result, related)
            if llm_result is not None:
                result, verifier = llm_result, "rules+llm"
        evidence = result.evidence or [e for nc in ex.number_checks for e in nc.evidence] or ex.code_facts[:3]
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict=result.verdict,
            confidence=round(result.confidence, 2),
            rationale=result.rationale.strip(),
            rules_fired=sorted(set(result.rules)),
            evidence=_dedupe(evidence),
            risk_ids=[r.finding_id for r in related],
            verifier=verifier,
        )

    def _llm_verify(self, claim: Claim, plan: TracePlan, ex: ExecutionEvidence, rule: RuleResult, related: Sequence[RiskFinding]) -> Optional[RuleResult]:
        cells_text = "\n\n".join(self.nb.cell_text(i, max_output_chars=3000) for i in plan.target_cells)
        data = self.llm.complete_json(
            VERIFIER_SYSTEM,
            verifier_user_prompt(_claim_payload(claim), rule.as_dict(), cells_text, [f"{r.finding_id} [{r.severity}] {r.message}" for r in related]),
            max_tokens=1200,
        )
        if not isinstance(data, dict) or data.get("verdict") not in VERDICTS:
            return None
        verdict = data["verdict"]
        if verdict == "UNVERIFIABLE":
            return RuleResult("UNVERIFIABLE", rule.confidence, rule.rationale + f" LLM verifier: {data.get('rationale', '')}", rule.rules + ["LLM-UNVERIFIABLE"], rule.evidence)
        citations = []
        for cit in data.get("citations") or []:
            try:
                cell = self.nb.get(int(cit.get("cell_index")))
            except (TypeError, ValueError, AttributeError):
                continue
            quote = _norm_ws(str(cit.get("quote") or ""))
            if cell is None or not quote or cell.cell_index not in plan.target_cells:
                continue
            in_code = quote in _norm_ws(cell.source_code)
            in_out = quote in _norm_ws(cell.output_text)
            if in_code or in_out:
                citations.append(EvidenceItem(cell.cell_index, "output" if in_out else "code", quote[:200], None, "", "match" if verdict == "SUPPORTED" else "mismatch"))
        if not citations:
            return None  # uncited verdicts are discarded
        untraced = [n for n, nc in zip(claim.numbers, ex.number_checks) if nc.status not in ("match", "derived")]
        if verdict == "SUPPORTED" and untraced:
            quoted_nums = [m for c in citations for m in extract_numbers(c.snippet)]
            if not all(any(values_match(n, q) for q in quoted_nums) for n in untraced):
                return None  # LLM cannot "support" numbers that no cited quote contains
        try:
            conf = max(0.0, min(float(data.get("confidence", 0.6)), 0.75))
        except (TypeError, ValueError):
            conf = 0.6
        return RuleResult(verdict, conf, f"LLM-VERIFIER (cited): {data.get('rationale', '')}", rule.rules + ["LLM-CITED"], citations)


def _dedupe(items: Sequence[EvidenceItem]) -> List[EvidenceItem]:
    seen, out = set(), []
    for e in items:
        key = (e.cell_index, e.location, e.snippet, e.relation)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


# --------------------------------------------------------------------------- orchestration
def _assign_ids(findings: List[RiskFinding], prefix: str, start: int = 1) -> None:
    for i, f in enumerate(findings, start=start):
        f.finding_id = f"{prefix}{i}"


class AuditPipeline:
    def __init__(self, llm: Optional[LLMClient] = None, threshold_tolerance: float = 0.005, findings: bool = False) -> None:
        self.llm = llm
        self.tol = threshold_tolerance
        self.findings = findings

    def _build_findings(self, nb, index, observations, figures, risks, planner, executor, verifier, warnings):
        builder = FindingsBuilder(nb, index, observations, figures, risks)
        findings = builder.build()
        if self.llm is not None:
            for f, claim in builder.llm_candidates(self.llm, findings):
                claim.claim_id = f"LF{len(findings) + 1}"
                plan = planner.heuristic_plan(claim)
                plan.target_cells = sorted(set(plan.target_cells) | set(f.cells))  # always look at the cited cell
                ex = executor.execute(claim, plan)
                v = verifier.verify(claim, plan, ex)
                if builder.accept_llm(f, v):
                    findings.append(f)
        for i, f in enumerate(findings, start=1):
            f.finding_id = f"FD{i}"
            builder.rate(f)
        warnings += builder.warnings
        return findings, build_gaps(nb, index, risks, observations, figures)

    def run(self, nb: Notebook, claims: List[Claim], artifact_paths: Sequence[str] = (), warnings: Optional[List[str]] = None) -> AuditResult:
        warnings = list(warnings or [])
        index = CodeIndex(nb)
        observations = extract_observations(nb)

        risk_findings = RiskAuditor(nb, index, observations, claims, self.tol).run()
        figures = FigureAuditor(nb, index, claims)
        figure_findings = figures.run()
        _assign_ids(risk_findings, "R")
        _assign_ids(figure_findings, "F")
        pre_risks = risk_findings + figure_findings

        planner = PlannerAgent(nb, observations, figures, self.llm)
        executor = ExecutorAgent(nb, observations, figures, self.llm)
        verifier = VerifierAgent(nb, index, pre_risks, self.llm)

        plans = planner.plan(claims)
        executions = [executor.execute(c, p) for c, p in zip(claims, plans)]
        verdicts = [verifier.verify(c, p, e) for c, p, e in zip(claims, plans, executions)]
        findings, gaps = ([], [])
        if self.findings:
            findings, gaps = self._build_findings(nb, index, observations, figures, pre_risks, planner, executor, verifier, warnings)

        strength_findings = ClaimStrengthAuditor().run(claims, executions, verdicts)
        evidence = EvidenceAuditor()
        evidence_findings = evidence.findings(claims, executions)
        _assign_ids(evidence_findings, "E")
        _assign_ids(strength_findings, "S")

        warnings += planner.warnings
        for ex in executions:
            warnings += [n for n in ex.notes if n.startswith("Rejected")]
        if self.llm is not None:
            warnings += self.llm.errors

        order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        all_risks = sorted(evidence_findings + pre_risks + strength_findings, key=lambda r: (order.get(r.severity, 9), r.finding_id[0], int(r.finding_id[1:] or 0)))
        return AuditResult(
            notebook_path=nb.path,
            artifact_paths=list(artifact_paths),
            generated_at=_dt.datetime.now().isoformat(timespec="seconds"),
            mode=self.llm.mode_label if self.llm else "heuristic",
            notebook_summary=notebook_summary(nb),
            claims=claims,
            plans=plans,
            executions=executions,
            verdicts=verdicts,
            risks=all_risks,
            evidence_map=evidence.build_map(claims, verdicts, executions),
            warnings=warnings,
            findings=findings,
            gaps=gaps,
        )


def run_audit(
    notebook_path: str,
    artifact_paths: Sequence[str] = (),
    provider: str = "auto",
    model: Optional[str] = None,
    include_notebook_markdown: Optional[bool] = None,
    threshold_tolerance: float = 0.005,
    llm: Optional[LLMClient] = None,
    findings: Optional[bool] = None,
) -> AuditResult:
    """End-to-end audit. With no API key (or provider='none') it runs fully offline.

    ``findings`` drafts "what the notebook found"; by default it is on when no write-up is given.
    """
    nb = load_notebook(notebook_path)
    client = llm if llm is not None else make_client(provider, model)
    if include_notebook_markdown is None:
        include_notebook_markdown = not artifact_paths
    if findings is None:
        findings = not artifact_paths
    claims, warnings = extract_claims(list(artifact_paths), nb, include_notebook_markdown, llm=client)
    return AuditPipeline(client, threshold_tolerance, findings=findings).run(nb, claims, artifact_paths, warnings)
