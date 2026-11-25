"""Core data structures shared across the parser, extractor, agents and auditors."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", "METHODOLOGICAL_RISK")
CATEGORIES = ("numeric", "trend", "methodological", "qualitative")
STRENGTHS = ("MEASURED", "DERIVED", "INTERPRETATION")
SEVERITIES = ("high", "medium", "low", "info")


# --------------------------------------------------------------------------- notebook
@dataclass
class ImageInfo:
    mime: str
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None

    def describe(self) -> str:
        dims = f" {self.width}x{self.height}" if self.width and self.height else ""
        return f"<{self.mime}{dims}, {self.size_bytes / 1024:.1f} KB>"


@dataclass
class CellOutput:
    output_type: str  # stream | execute_result | display_data | error
    text: str = ""  # best plain-text representation
    stream_name: Optional[str] = None  # stdout | stderr
    html_tables: List[List[List[str]]] = field(default_factory=list)
    images: List[ImageInfo] = field(default_factory=list)
    error_name: Optional[str] = None
    traceback: str = ""


@dataclass
class Cell:
    cell_index: int  # 1-based position in the notebook ("Cell 1" is the first cell)
    cell_type: str  # code | markdown | raw
    source_code: str
    execution_count: Optional[int]
    outputs: List[CellOutput] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"Cell {self.cell_index}"

    @property
    def output_text(self) -> str:
        parts: List[str] = []
        for out in self.outputs:
            if out.text:
                parts.append(out.text.rstrip("\n"))
            for img in out.images:
                parts.append(img.describe())
        return "\n".join(parts)

    @property
    def images(self) -> List[ImageInfo]:
        return [img for out in self.outputs for img in out.images]

    @property
    def has_error(self) -> bool:
        return any(o.output_type == "error" for o in self.outputs)


@dataclass
class Notebook:
    path: str
    cells: List[Cell]
    language: str = "python"
    nbformat: int = 4

    @property
    def code_cells(self) -> List[Cell]:
        return [c for c in self.cells if c.cell_type == "code"]

    @property
    def markdown_cells(self) -> List[Cell]:
        return [c for c in self.cells if c.cell_type == "markdown"]

    def get(self, index: int) -> Optional[Cell]:
        if 1 <= index <= len(self.cells):
            return self.cells[index - 1]
        return None

    def cell_text(self, index: int, max_output_chars: Optional[int] = None) -> str:
        cell = self.get(index)
        if cell is None:
            return ""
        return render_cell(cell, max_output_chars)

    def to_linear_text(self, max_output_chars: Optional[int] = 4000, include_markdown: bool = True) -> str:
        blocks = []
        for cell in self.cells:
            if cell.cell_type != "code" and not include_markdown:
                continue
            blocks.append(render_cell(cell, max_output_chars))
        return "\n\n".join(blocks)


def render_cell(cell: Cell, max_output_chars: Optional[int] = None) -> str:
    kind = {"code": "Code", "markdown": "Markdown"}.get(cell.cell_type, cell.cell_type.title())
    exec_tag = f" (In [{cell.execution_count}])" if cell.execution_count is not None else ""
    lines = [f"[Cell {cell.cell_index} - {kind}]{exec_tag}", cell.source_code.rstrip()]
    if cell.cell_type == "code" and cell.outputs:
        out = cell.output_text
        if max_output_chars is not None and len(out) > max_output_chars:
            out = out[:max_output_chars] + f"\n... [truncated {len(out) - max_output_chars} chars]"
        lines.append(f"[Cell {cell.cell_index} - Output]")
        lines.append(out)
    return "\n".join(lines)


# --------------------------------------------------------------------------- numbers & claims
@dataclass
class NumberMention:
    raw: str
    value: float  # value as written (85% -> 85.0, 3k -> 3000)
    is_percent: bool
    decimals: int
    qualifier: Optional[str] = None  # approx | gt | lt
    start: int = 0
    end: int = 0
    before: str = ""  # text between the previous number and this one
    after: str = ""  # text up to the next number
    kinds: List[str] = field(default_factory=list)  # metric kinds (accuracy, count, ...)
    qualifiers: List[str] = field(default_factory=list)  # train | test | val | baseline ...


@dataclass
class Claim:
    claim_id: str
    text: str  # self-contained (pronouns resolved) statement
    original_text: str
    source: str  # e.g. "slides.md:12" or "notebook:Cell 5"
    context: str  # section / slide heading
    category: str  # numeric | trend | methodological | qualitative
    tags: List[str] = field(default_factory=list)
    numbers: List[NumberMention] = field(default_factory=list)
    direction: Optional[str] = None  # increase | decrease | stable
    extractor: str = "heuristic"


# --------------------------------------------------------------------------- evidence
@dataclass
class Observation:
    """A number printed by (or literally written in) a notebook cell."""

    cell_index: int
    location: str  # output | code
    value: float
    raw: str
    is_percent: bool
    decimals: int
    label: str
    kinds: List[str]
    qualifiers: List[str]
    line: str
    order: int = 0
    source: str = "heuristic"  # heuristic | llm


@dataclass
class EvidenceItem:
    cell_index: int
    location: str  # output | code
    snippet: str
    value: Optional[float] = None
    label: str = ""
    relation: str = "context"  # match | mismatch | derived | weak_match | context | violation


@dataclass
class NumberCheck:
    raw: str
    status: str  # match | derived | mismatch | weak_match | untraced
    evidence: List[EvidenceItem] = field(default_factory=list)
    note: str = ""


@dataclass
class SeriesEvidence:
    cell_index: int
    label: str
    values: List[float]
    direction: Optional[str]
    monotonic: bool
    source: str  # output | plot_literal


@dataclass
class TracePlan:
    claim_id: str
    target_cells: List[int]
    rationale: str
    planner: str = "heuristic"


@dataclass
class ExecutionEvidence:
    claim_id: str
    number_checks: List[NumberCheck] = field(default_factory=list)
    series: List[SeriesEvidence] = field(default_factory=list)
    code_facts: List[EvidenceItem] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    executor: str = "heuristic"


@dataclass
class RiskFinding:
    finding_id: str
    engine: str  # evidence | risk | strength | figure
    check: str
    severity: str  # high | medium | low | info
    message: str
    cell_indices: List[int] = field(default_factory=list)
    detail: str = ""
    claim_ids: List[str] = field(default_factory=list)


@dataclass
class ClaimVerdict:
    claim_id: str
    verdict: str
    confidence: float
    rationale: str
    rules_fired: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    risk_ids: List[str] = field(default_factory=list)
    strength: Optional[str] = None
    traced: bool = False
    does_not_prove: Optional[str] = None
    verifier: str = "rules"


@dataclass
class AuditResult:
    notebook_path: str
    artifact_paths: List[str]
    generated_at: str
    mode: str  # heuristic | llm:<provider>/<model>
    notebook_summary: Dict[str, Any]
    claims: List[Claim]
    plans: List[TracePlan]
    executions: List[ExecutionEvidence]
    verdicts: List[ClaimVerdict]
    risks: List[RiskFinding]
    evidence_map: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def verdict_for(self, claim_id: str) -> Optional[ClaimVerdict]:
        return next((v for v in self.verdicts if v.claim_id == claim_id), None)

    def claim(self, claim_id: str) -> Optional[Claim]:
        return next((c for c in self.claims if c.claim_id == claim_id), None)

    def summary(self) -> Dict[str, Any]:
        counts = {v: 0 for v in VERDICTS}
        for v in self.verdicts:
            counts[v.verdict] = counts.get(v.verdict, 0) + 1
        sev = {s: 0 for s in SEVERITIES}
        for r in self.risks:
            sev[r.severity] = sev.get(r.severity, 0) + 1
        return {"claims": len(self.claims), "verdicts": counts, "risks_by_severity": sev}

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["summary"] = self.summary()
        return data
