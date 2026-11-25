"""Extract atomic, self-contained factual claims from slides / papers / markdown cells."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from .models import Claim, Notebook
from .textnum import extract_numbers, kinds_of_text, tokenize, values_match

if TYPE_CHECKING:  # pragma: no cover
    from .llm import LLMClient

TREND_UP = {
    "increase", "increased", "increases", "increasing", "rise", "rose", "risen", "rising", "rises", "grow", "grew",
    "grows", "growing", "improve", "improved", "improves", "improving", "gain", "gained", "boost", "boosted",
    "climb", "climbed", "climbs", "upward", "outperform", "outperforms", "outperformed", "higher",
}
TREND_DOWN = {
    "decrease", "decreased", "decreases", "decreasing", "decline", "declined", "declines", "declining", "drop",
    "dropped", "drops", "fall", "fell", "falls", "falling", "reduce", "reduced", "reduces", "reduction", "lower",
    "downward", "degrade", "degraded", "degrades", "shrink", "shrank", "diminish", "diminished", "worse",
}
TREND_STABLE = {"plateau", "plateaued", "plateaus", "stable", "stabilized", "stabilised", "flat", "unchanged", "converged", "saturated"}
MONOTONIC_WORDS = {"steady", "steadily", "consistently", "consistent", "monotonic", "monotonically", "continuously", "every"}
NEGATIONS = {"not", "no", "never", "didn", "doesn", "isn", "wasn", "without"}

METHOD_PATTERNS = [
    r"\bsplit", r"stratif", r"cross[- ]?valid", r"\bk[- ]fold", r"\d+[- ]fold", r"held[- ]out", r"hold[- ]?out",
    r"baseline", r"majority[- ]class", r"\bchance\b", r"random (?:seed|state)", r"\bseed", r"standardi[sz]",
    r"normali[sz]", r"\bscal(?:ed|ing|er)\b", r"imput", r"preprocess", r"de-?duplicat", r"duplicates",
    r"tokeni[sz]", r"lemmati[sz]", r"stemm", r"stop ?words", r"vectori[sz]", r"tf-?idf", r"oversampl",
    r"undersampl", r"\bsmote\b", r"shuffl", r"train/test", r"training set", r"test set", r"validation set",
    r"missing values", r"leak",
]
METHOD_RE = re.compile("|".join(METHOD_PATTERNS), re.I)
SPLIT_RATIO_RE = re.compile(r"\b\d{1,2}\s*[/:-]\s*\d{1,2}\b")
METHOD_PARAM_TOKENS = {"split", "fold", "folds", "seed", "ratio", "held", "holdout", "stratified"}

INTERPRETIVE_RE = re.compile(
    r"\b(demonstrat\w*|prove[sd]?|proof|clearly|shows? that|suggest\w*|indicat\w*|understand\w*|learns?|learned|"
    r"captur\w*|robust\w*|generali[sz]\w*|caus\w*|because|due to|significant\w*|reliabl\w*|effective\w*|"
    r"superior|strong(?:ly)?|well[- ]separated|separat\w+|meaningful|evidence|confirm\w*|reveal\w*|"
    r"outperform\w*|better|best|state[- ]of[- ]the[- ]art)\b",
    re.I,
)

VERBS = {
    "is", "was", "are", "were", "has", "had", "have", "achieved", "achieves", "achieve", "reached", "reaches",
    "shows", "showed", "show", "contains", "contained", "contain", "scored", "scores", "yields", "yielded",
    "gave", "gives", "obtained", "obtains", "reported", "reports", "comprises", "includes", "included",
    "produced", "produces", "used", "uses", "trained", "converged", "remained", "stayed",
} | TREND_UP | TREND_DOWN | TREND_STABLE
PRONOUNS = {"it", "this", "that", "they", "these", "those", "its", "their"}
_ABBREV = ["e.g.", "i.e.", "vs.", "etc.", "Fig.", "fig.", "approx.", "et al.", "No.", "Eq.", "cf."]


@dataclass
class _Unit:
    text: str
    source: str
    context: str


# --------------------------------------------------------------------------- markdown -> units
def _strip_inline_md(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("$", "")
    text = re.sub(r"\\(alpha|le|leq|ge|geq|approx|times)", lambda m: {"alpha": "α", "le": "<=", "leq": "<=", "ge": ">=", "geq": ">=", "approx": "~", "times": "x"}[m.group(1)], text)
    return " ".join(text.split())


def _split_sentences(text: str) -> List[str]:
    protected = text
    for i, ab in enumerate(_ABBREV):
        protected = protected.replace(ab, f"\x00{i}\x00")
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'0-9~])", protected)
    out = []
    for p in parts:
        for i, ab in enumerate(_ABBREV):
            p = p.replace(f"\x00{i}\x00", ab)
        p = p.strip()
        if p:
            out.append(p)
    return out


def markdown_units(text: str, source_name: str, line_offset: int = 0) -> List[_Unit]:
    units: List[_Unit] = []
    headings: List[str] = []
    in_fence = False
    in_comment = False
    table_header: Optional[List[str]] = None
    para: List[Tuple[int, str]] = []

    def flush() -> None:
        if not para:
            return
        lineno = para[0][0]
        joined = _strip_inline_md(" ".join(t for _, t in para))
        for sent in _split_sentences(joined):
            units.append(_Unit(sent, f"{source_name}:{lineno}", " > ".join(headings)))
        para.clear()

    for i, raw in enumerate(text.splitlines(), start=1 + line_offset):
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if "<!--" in stripped:
            in_comment = "-->" not in stripped
            continue
        if in_comment:
            in_comment = "-->" not in stripped
            continue
        if not stripped or re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush()
            table_header = None
            if re.fullmatch(r"-{3,}", stripped or "x"):
                headings = []  # slide separator resets slide context
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        bold = None if m else re.fullmatch(r"(\*\*|__)([^*_]+)\1:?", stripped)  # a bold-only line acts as a heading
        if m or bold:
            flush()
            level = len(m.group(1)) if m else len(headings) + 1
            title = m.group(2) if m else bold.group(2)
            headings = headings[: level - 1] + [_strip_inline_md(title)]
            continue
        if stripped.startswith("|"):
            flush()
            cells = [_strip_inline_md(c) for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c):
                continue
            if table_header is None:
                table_header = cells
                continue
            pairs = [f"{h}: {v}" for h, v in zip(table_header, cells) if v]
            if pairs:
                units.append(_Unit(", ".join(pairs), f"{source_name}:{i}", " > ".join(headings)))
            continue
        bullet = re.match(r"^(?:[-*+]|\d+[.)])\s+(.*)", stripped)
        if bullet:
            flush()
            para.append((i, bullet.group(1)))
            flush()
            continue
        if stripped.startswith(">"):
            stripped = stripped.lstrip("> ")
        para.append((i, stripped))
    flush()
    return units


# --------------------------------------------------------------------------- unit -> claims
def _first_verb_index(tokens: List[str]) -> int:
    for i, t in enumerate(tokens):
        if t in VERBS:
            return i
    return -1


def _subject_of(sentence: str) -> Optional[str]:
    words = sentence.split()
    toks = [w.strip(",.;:()").lower() for w in words]
    vi = _first_verb_index(toks)
    if 0 < vi <= 8 and toks[0] not in PRONOUNS:
        return " ".join(words[:vi]).strip(",;:")
    return None


def _atomic_parts(sentence: str) -> List[str]:
    parts: List[str] = []
    for chunk in re.split(r";\s+", sentence):
        pieces = re.split(r",?\s+(?:and|while|whereas|but)\s+", chunk)
        if len(pieces) > 1 and all(extract_numbers(p) and _first_verb_index(tokenize(p)) >= 0 for p in pieces):
            subject = _subject_of(pieces[0])
            for k, p in enumerate(pieces):
                if k > 0 and subject and _first_verb_index(tokenize(p)) == 0:
                    p = f"{subject} {p}"
                parts.append(p.strip())
        else:
            parts.append(chunk.strip())
    return [p for p in parts if p]


def _resolve_pronouns(sentence: str, last_subject: Optional[str], context: str) -> str:
    m = re.match(r"^(It|They|This|These|That|Those|Its|Their)\b(\s+\w+)?", sentence)
    if not m:
        return sentence
    word, nxt = m.group(1), (m.group(2) or "").strip().lower()
    referent = last_subject or context
    if not referent:
        return sentence
    if word in ("Its", "Their"):
        return f"{referent}'s{sentence[len(word):]}"
    if word in ("This", "These", "That", "Those") and nxt and nxt not in VERBS:
        # "This model achieves..." -> "This model (<referent>) achieves..."
        head = m.group(0)
        return f"{head} ({referent}){sentence[len(head):]}"
    return f"{referent}{sentence[len(word):]}"


def _direction(text: str) -> Tuple[Optional[str], List[str]]:
    toks = tokenize(text)
    tags: List[str] = []
    direction = None
    for i, t in enumerate(toks):
        d = "increase" if t in TREND_UP else "decrease" if t in TREND_DOWN else "stable" if t in TREND_STABLE else None
        if not d:
            continue
        window = toks[max(0, i - 3) : i]
        if any(w in NEGATIONS or w.endswith("n't") for w in window):
            tags.append("negated_trend")
            continue
        if t in {"higher", "lower", "better", "worse", "outperform", "outperforms", "outperformed"}:
            tags.append("comparison")
        direction = d
        break
    if direction and set(toks) & MONOTONIC_WORDS:
        tags.append("monotonic")
    return direction, tags


def _numbers_are_method_params(text: str, numbers) -> bool:
    if SPLIT_RATIO_RE.search(text):
        return True
    for n in numbers:
        ctx = set(tokenize(n.before)[-4:]) | set(tokenize(n.after)[:3])
        if not ctx & METHOD_PARAM_TOKENS:
            return False
    return True


def categorize(text: str, numbers) -> Tuple[Optional[str], List[str], Optional[str]]:
    direction, tags = _direction(text)
    has_method = bool(METHOD_RE.search(text))
    if numbers:
        tags.append("numeric")
    if direction:
        tags.append("trend")
    if has_method:
        tags.append("methodological")
    if INTERPRETIVE_RE.search(text):
        tags.append("interpretive")
    if has_method and (not numbers or _numbers_are_method_params(text, numbers)):
        return "methodological", tags, direction
    if numbers:
        return "numeric", tags, direction
    if direction:
        return "trend", tags, direction
    if has_method:
        return "methodological", tags, direction
    if "interpretive" in tags:
        return "qualitative", tags, direction
    return None, tags, direction


STRUCTURE_RE = re.compile(r"^(step|task|part|question|q|problem|exercise|section|chapter|lab|assignment|week|module|phase|stage|appendix|slide)\s*\d", re.I)


def _is_candidate(text: str) -> bool:
    if text.endswith("?") or len(text.split()) < 3:
        return False
    if re.match(r"^(thank|thanks|questions|agenda|outline|references|q&a)\b", text, re.I):
        return False
    if STRUCTURE_RE.match(text):  # "Step 2: Task 1 — EDA" is document structure, not a finding
        return False
    return True


class ClaimExtractor:
    """Heuristic extractor, optionally refined by an LLM (decontextualisation + atomisation)."""

    def __init__(self, llm: Optional["LLMClient"] = None) -> None:
        self.llm = llm
        self.warnings: List[str] = []

    # public API -------------------------------------------------------------
    def from_markdown(self, text: str, source_name: str) -> List[Claim]:
        return self._units_to_claims(markdown_units(text, source_name))

    def from_file(self, path) -> List[Claim]:
        path = Path(path)
        return self.from_markdown(path.read_text(encoding="utf-8"), path.name)

    def from_notebook(self, nb: Notebook) -> List[Claim]:
        units: List[_Unit] = []
        for cell in nb.markdown_cells:
            for u in markdown_units(cell.source_code, f"notebook:Cell {cell.cell_index}"):
                u.source = f"notebook:Cell {cell.cell_index}"
                units.append(u)
        return self._units_to_claims(units)

    # internals --------------------------------------------------------------
    def _units_to_claims(self, units: List[_Unit]) -> List[Claim]:
        claims = self._heuristic(units)
        if self.llm is not None and units:
            refined = self._llm_refine(units)
            if refined is not None:
                claims = refined
        for i, c in enumerate(claims, start=1):
            c.claim_id = f"C{i}"
        return claims

    def _heuristic(self, units: List[_Unit]) -> List[Claim]:
        claims: List[Claim] = []
        last_subject: Optional[str] = None
        last_context = None
        for unit in units:
            if unit.context != last_context:
                last_subject, last_context = None, unit.context
            for k, part in enumerate(_atomic_parts(unit.text)):
                if not _is_candidate(part):
                    continue
                resolved = _resolve_pronouns(part, last_subject, unit.context)
                claim = self._make_claim(resolved, unit.text, unit.source, unit.context)
                subj = _subject_of(resolved) if k == 0 else None  # "It" refers to the main clause's subject
                if subj:
                    last_subject = subj
                if claim:
                    claims.append(claim)
        return claims

    @staticmethod
    def _make_claim(text: str, original: str, source: str, context: str, extractor: str = "heuristic") -> Optional[Claim]:
        numbers = extract_numbers(text, exclude_refs=True)
        if numbers:
            sentence_kinds = kinds_of_text(text)
            for n in numbers:
                if not n.kinds and len(sentence_kinds) == 1:
                    n.kinds = sorted(sentence_kinds)
        category, tags, direction = categorize(text, numbers)
        if category is None:
            return None
        return Claim(
            claim_id="",
            text=text.rstrip(" ."),
            original_text=original,
            source=source,
            context=context,
            category=category,
            tags=sorted(set(tags)),
            numbers=numbers,
            direction=direction,
            extractor=extractor,
        )

    def _llm_refine(self, units: List[_Unit]) -> Optional[List[Claim]]:
        from .prompts import CLAIM_SYSTEM, claim_user_prompt

        payload = [{"unit_id": i, "context": u.context, "text": u.text} for i, u in enumerate(units)]
        data = self.llm.complete_json(CLAIM_SYSTEM, claim_user_prompt(payload), max_tokens=4000) if self.llm else None
        if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
            self.warnings.append("LLM claim extraction failed; using heuristic claims.")
            return None
        claims: List[Claim] = []
        for item in data["claims"]:
            try:
                unit = units[int(item["unit_id"])]
                text = str(item["claim"]).strip()
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            # Anti-hallucination: every number in the rewritten claim must exist in the source unit.
            src_numbers = extract_numbers(unit.text)
            new_numbers = extract_numbers(text, exclude_refs=True)
            if any(not any(values_match(n, s) and values_match(s, n) for s in src_numbers) for n in new_numbers):
                self.warnings.append(f"Dropped LLM claim with numbers not in source: {text!r}")
                continue
            claim = self._make_claim(text, unit.text, unit.source, unit.context, extractor="llm")
            if claim is None:
                continue
            cat = str(item.get("category", "")).lower()
            if cat in ("numeric", "trend", "methodological", "qualitative") and not (cat == "numeric" and not claim.numbers):
                claim.category = cat
            claims.append(claim)
        return claims or None


def extract_claims(paths: List[str], notebook: Optional[Notebook] = None, include_notebook_markdown: bool = False, llm=None) -> Tuple[List[Claim], List[str]]:
    extractor = ClaimExtractor(llm=llm)
    units: List[_Unit] = []
    for p in paths:
        path = Path(p)
        units.extend(markdown_units(path.read_text(encoding="utf-8"), path.name))
    if notebook is not None and include_notebook_markdown:
        for cell in notebook.markdown_cells:
            for u in markdown_units(cell.source_code, f"notebook:Cell {cell.cell_index}"):
                u.source = f"notebook:Cell {cell.cell_index}"
                units.append(u)
    claims = extractor._units_to_claims(units)
    return claims, extractor.warnings
