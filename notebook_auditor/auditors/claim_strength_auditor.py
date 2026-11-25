"""Claim epistemic-modesty auditor: MEASURED / DERIVED / INTERPRETATION + "What this does not prove"."""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from ..claim_extractor import INTERPRETIVE_RE
from ..models import Claim, ClaimVerdict, ExecutionEvidence, RiskFinding

CERTAINTY_RE = re.compile(r"\b(prove[sd]?|demonstrat\w*|clearly|confirm\w*|establish\w*|definitively|undeniabl\w*)\b", re.I)
# Language that goes beyond a reported number (a p-value or "outperforms" alone is a derived statement).
STRONG_INTERP_RE = re.compile(r"\b(demonstrat\w*|prove[sd]?|clearly|understand\w*|learn(?:s|ed)?|captur\w*|robust\w*|generali[sz]\w*|caus\w*|because|due to|reliabl\w*|separat\w+|meaningful)\b", re.I)
ARITHMETIC_RE = re.compile(r"\b(improv\w*|differen\w*|gap|times|relative|ratio|average|mean of|higher by|lower by|percentage points|pp|gain of|drop of|reduction of|increase of|decrease of)\b", re.I)

# (pattern, limit) — first match wins; ordered from most to least specific.
LIMITS: List[Tuple[str, str]] = [
    (r"t-?sne|umap|2-?d projection|embedding (?:plot|space)|cluster",
     "Visual clustering in a 2-D t-SNE/UMAP projection does not prove statistical separation of the classes; "
     "projection distances and cluster sizes are not faithful, and layouts depend on perplexity and random seed."),
    (r"p\s*[=<]|p-value|statistically|significan",
     "Statistical significance does not establish effect size or practical importance, and is not robust to "
     "multiple comparisons or analysis choices unless those are reported."),
    (r"correlat|associat|relationship between",
     "A correlation does not prove a causal relationship and may be driven by confounders or outliers."),
    (r"attention|shap|feature importance|saliency|important features|coefficients?",
     "Feature importance / attention weights describe what the model used, not a causal mechanism in the data."),
    (r"outperform|better than|beats|superior|exceeds|surpass|improv",
     "A higher score on one split does not prove a real improvement unless the gap exceeds run-to-run variance "
     "(multiple seeds, confidence intervals or a paired significance test)."),
    (r"understand|learn(?:s|ed)?|capture|reason|knows|aware",
     "Behavioural metrics do not prove the model internally 'understands' or has 'learned' the concept; "
     "it may exploit surface cues or dataset artifacts."),
    (r"generali[sz]|robust|real[- ]world|in the wild|deploy",
     "Performance on this held-out split does not prove generalisation to other domains, time periods or populations."),
    (r"accuracy|f1|precision|recall|auc|score",
     "A single held-out score does not prove the model generalises beyond this data split; no variance or "
     "confidence interval is reported, and without a baseline the score's difficulty is unknown."),
    (r"increas|decreas|trend|declin|rose|fell|drop|grow|steady|plateau|converg",
     "A trend over the observed range does not prove it continues outside that range or is caused by the "
     "manipulated variable."),
    (r"documents|samples|rows|records|examples|participants|dataset",
     "A sample count does not prove the data are representative, deduplicated or correctly labelled."),
    (r"split|stratif|cross-?valid|fold|held[- ]out|baseline|scal|normali",
     "Stating a procedure does not prove it was applied correctly (e.g. transforms fitted only on training data)."),
]


def does_not_prove(claim: Claim) -> Optional[str]:
    for pattern, text in LIMITS:
        if re.search(pattern, claim.text, re.I):
            return text
    if claim.category in ("qualitative", "trend") or "interpretive" in claim.tags:
        return "This interpretation goes beyond what the printed outputs directly show; treat it as a hypothesis."
    return None


class ClaimStrengthAuditor:
    engine = "strength"

    def assess(self, claim: Claim, execution: Optional[ExecutionEvidence], verdict: Optional[ClaimVerdict]) -> Tuple[str, bool]:
        statuses = [nc.status for nc in (execution.number_checks if execution else [])]
        traced = bool(statuses) and all(s in ("match", "derived") for s in statuses)
        if claim.category == "qualitative" or (INTERPRETIVE_RE.search(claim.text) and claim.category != "numeric"):
            return "INTERPRETATION", traced
        if claim.category == "numeric" and STRONG_INTERP_RE.search(claim.text):
            return "INTERPRETATION", traced
        if "derived" in statuses or (claim.numbers and ARITHMETIC_RE.search(claim.text)):
            return "DERIVED", traced
        if claim.category == "trend" or (claim.direction and "comparison" not in claim.tags):
            supported = bool(verdict and verdict.verdict == "SUPPORTED")
            return ("DERIVED" if supported else "INTERPRETATION"), supported
        if claim.category == "methodological":
            return "MEASURED", bool(verdict and verdict.verdict in ("SUPPORTED", "METHODOLOGICAL_RISK"))
        if claim.numbers:
            return "MEASURED", traced
        return "INTERPRETATION", False

    def run(self, claims: Sequence[Claim], executions: Sequence[ExecutionEvidence], verdicts: Sequence[ClaimVerdict]) -> List[RiskFinding]:
        ex_by = {e.claim_id: e for e in executions}
        findings: List[RiskFinding] = []
        for v in verdicts:
            claim = next((c for c in claims if c.claim_id == v.claim_id), None)
            if claim is None:
                continue
            strength, traced = self.assess(claim, ex_by.get(claim.claim_id), v)
            v.strength, v.traced = strength, traced
            v.does_not_prove = does_not_prove(claim)
            certainty = CERTAINTY_RE.search(claim.text)
            if strength == "INTERPRETATION" and certainty:
                findings.append(
                    RiskFinding("", self.engine, "overclaiming", "medium",
                                f"{claim.claim_id} uses certainty language ('{certainty.group(0)}') for an interpretation",
                                [], v.does_not_prove or "", [claim.claim_id])
                )
        return findings
