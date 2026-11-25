"""Number extraction, metric-label vocabulary and tolerant value matching.

Everything here is deterministic; the LLM agents are layered on top of it and
their output is always re-checked with these functions.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from .models import NumberMention

_TOKEN_RE = re.compile(r"[a-z0-9²α]+")
_FSCORE_RE = re.compile(r"\bf[-_ ]?(?:1|score|measure)\b", re.I)


_PVALUE_RE = re.compile(r"\bp[-_ ]?val(?:ue)?s?\b|\bp\s*(?=[=<>≤])", re.I)


def tokenize(text: str) -> List[str]:
    text = _FSCORE_RE.sub(" f1 ", text)
    # A bare "p" is only a p-value in "p = 0.04", "p < .05" or "p-value"; "P(→L2 | L1)" is a probability.
    text = _PVALUE_RE.sub(" pvalue ", text)
    return _TOKEN_RE.findall(text.lower())


STOPWORDS = frozenset(
    """a an the of to in on for with by and or is are was were be been being this that these those it its
    as at from into than then there their our we us you your they them he she his her not no yes do does did
    done has have had can could would should will may might also which who whom whose what when where why how
    all any each both more most other some such only own same so too very just over under about after before
    again further once here out up down off using use used per vs via""".split()
)

METRIC_SYNONYMS = {
    "accuracy": {"accuracy", "acc", "accuracies"},
    "f1": {"f1"},
    "precision": {"precision"},
    "recall": {"recall", "sensitivity", "tpr"},
    "auc": {"auc", "roc", "auroc"},
    "loss": {"loss", "losses", "nll", "crossentropy"},
    "rmse": {"rmse"},
    "mse": {"mse"},
    "mae": {"mae"},
    "r2": {"r2", "r²", "rsquared"},
    "p_value": {"pvalue", "pval", "pvalues"},
    "correlation": {"correlation", "corr", "pearson", "spearman", "rho"},
    "count": {
        "documents", "document", "docs", "doc", "rows", "row", "samples", "sample", "records", "examples",
        "observations", "instances", "entries", "items", "n", "count", "len", "total", "sentences", "tweets",
        "reviews", "texts", "articles", "participants", "users", "images", "shape", "utterances", "posts",
        "comments", "messages", "files", "papers", "abstracts", "transcripts", "respondents",
    },
    "epochs": {"epoch", "epochs"},
    "features": {"features", "feature", "dimensions", "dimension", "dims", "columns", "cols", "vocabulary", "vocab"},
    "classes": {"classes", "categories", "clusters", "topics"},
    "silhouette": {"silhouette"},
    "perplexity": {"perplexity"},
    "bleu": {"bleu"},
    "rouge": {"rouge"},
    "threshold": {"threshold", "cutoff", "alpha", "α"},
}
TOKEN_TO_KIND = {tok: kind for kind, toks in METRIC_SYNONYMS.items() for tok in toks}

# Metrics bounded in [0, 1] (or [0, 100] %) - used by the figure & threshold checks.
BOUNDED_KINDS = {"accuracy", "f1", "precision", "recall", "auc", "r2", "silhouette"}
EVAL_KINDS = {"accuracy", "f1", "precision", "recall", "auc", "rmse", "mse", "mae", "r2", "loss", "bleu", "rouge"}

QUALIFIER_TOKENS = {
    "train": "train", "training": "train", "trainset": "train",
    "test": "test", "testing": "test", "heldout": "test", "holdout": "test", "unseen": "test",
    "val": "val", "validation": "val", "dev": "val", "valid": "val",
    "baseline": "baseline", "majority": "baseline", "dummy": "baseline", "chance": "baseline",
    "macro": "macro", "micro": "micro", "weighted": "weighted",
}

REFERENCE_WORDS = {
    "figure", "fig", "table", "slide", "cell", "section", "step", "chapter", "page", "eq", "equation",
    "appendix", "version", "v", "line", "python", "gpt", "top", "layer", "item",
}
APPROX_WORDS = {"about", "approximately", "approx", "around", "roughly", "nearly", "almost", "circa", "ca"}
GT_PHRASES = ("more than", "over", "above", "exceeds", "exceeding", "at least", "greater than", "upwards of")
LT_PHRASES = ("less than", "under", "below", "fewer than", "at most", "up to")

NUM_RE = re.compile(
    r"(?<![A-Za-z_\d.])(?<![A-Za-z]-)"
    r"(?P<sign>-)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<suffix>\s?%|\s?percent\b|[kK]\b|M\b)?"
    r"(?![A-Za-z_\d])"
)


def kinds_of_tokens(tokens: Iterable[str]) -> Set[str]:
    return {TOKEN_TO_KIND[t] for t in tokens if t in TOKEN_TO_KIND}


def kinds_of_text(text: str) -> Set[str]:
    return kinds_of_tokens(tokenize(text))


def qualifiers_of_tokens(tokens: Iterable[str]) -> Set[str]:
    return {QUALIFIER_TOKENS[t] for t in tokens if t in QUALIFIER_TOKENS}


def content_tokens(text: str) -> Set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and not t.isdigit() and len(t) > 1}


def _is_num_token(tok: str) -> bool:
    return any(ch.isdigit() for ch in tok) and tok not in TOKEN_TO_KIND


def nearest_kinds(before: str, after: str, back: int = 6, fwd: int = 3) -> Tuple[Set[str], Set[str]]:
    """Return (kinds, qualifiers) closest to a number given the text around it."""
    btoks = tokenize(before)[::-1][:back]
    atoks = tokenize(after)[:fwd]
    best: Set[str] = set()
    best_d = 99
    for d, tok in enumerate(btoks, start=1):
        if _is_num_token(tok):
            break
        if tok in TOKEN_TO_KIND:
            best, best_d = {TOKEN_TO_KIND[tok]}, d
            break
    for d, tok in enumerate(atoks, start=1):
        if _is_num_token(tok):
            break
        if tok in TOKEN_TO_KIND:
            if d < best_d:
                best, best_d = {TOKEN_TO_KIND[tok]}, d
            elif d == best_d:
                best.add(TOKEN_TO_KIND[tok])
            break
    quals = qualifiers_of_tokens(btoks) | qualifiers_of_tokens(atoks)
    return best, quals


def extract_numbers(text: str, exclude_refs: bool = False) -> List[NumberMention]:
    matches = list(NUM_RE.finditer(text))
    mentions: List[NumberMention] = []
    for i, m in enumerate(matches):
        raw_num = m.group("num")
        suffix = (m.group("suffix") or "").strip()
        prev_end = matches[i - 1].end() if i > 0 else 0
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        before = text[prev_end : m.start()]
        after = text[m.end() : next_start]
        prev_words = tokenize(text[max(0, m.start() - 30) : m.start()])
        if exclude_refs and prev_words and prev_words[-1] in REFERENCE_WORDS:
            continue
        value = float(raw_num.replace(",", ""))
        if m.group("sign"):
            value = -value
        is_percent = suffix in ("%", "percent")
        if suffix in ("k", "K"):
            value *= 1_000
        elif suffix == "M":
            value *= 1_000_000
        decimals = len(raw_num.split(".")[1]) if "." in raw_num else 0
        lead = text[max(0, m.start() - 25) : m.start()].lower()
        qualifier = None
        tail = text[m.end() : m.end() + 4]
        if re.match(r"\s*[–—-]\s*\d", tail) or re.search(r"\d\s*[–—-]\s*$", text[max(0, m.start() - 4) : m.start()]):
            qualifier = "range"  # "Medium 3–6" defines a bin; it is not a measured value
        elif tail.startswith("+"):
            qualifier = "gt"
        elif lead.rstrip().endswith("~") or (prev_words and prev_words[-1] in APPROX_WORDS) or suffix in ("k", "K", "M"):
            qualifier = "approx"
        elif any(lead.rstrip().endswith(p) for p in GT_PHRASES) or lead.rstrip().endswith(">"):
            qualifier = "gt"
        elif any(lead.rstrip().endswith(p) for p in LT_PHRASES) or lead.rstrip().endswith("<"):
            qualifier = "lt"
        # A label never reaches across a list separator: in "t = 2.31, p = 0.04" the "p" belongs to 0.04 only.
        kinds, quals = nearest_kinds(re.split(r"[,;|]", before)[-1], re.split(r"[,;=:|(]", after)[0])
        mentions.append(
            NumberMention(
                raw=m.group(0).strip(),
                value=value,
                is_percent=is_percent,
                decimals=decimals,
                qualifier=qualifier,
                start=m.start(),
                end=m.end(),
                before=before,
                after=after,
                kinds=sorted(kinds),
                qualifiers=sorted(quals),
            )
        )
    return mentions


# --------------------------------------------------------------------------- matching
def _pairs(claim: NumberMention, obs) -> List[Tuple[float, float, Optional[int]]]:
    """Candidate (claim value, observed value, observed decimals) pairs in a common unit."""
    odec = obs.decimals
    if claim.is_percent:
        if obs.is_percent:
            return [(claim.value, obs.value, odec)]
        if abs(obs.value) <= 1.0:
            return [(claim.value, obs.value * 100, max(odec - 2, 0) if odec > 2 else None)]
        return [(claim.value, obs.value, odec)]  # e.g. printed "85.12" meaning percent
    if obs.is_percent:
        return [(claim.value, obs.value, odec), (claim.value, obs.value / 100, odec + 2)]
    return [(claim.value, obs.value, odec)]


def _close(c: float, o: float, claim: NumberMention, obs_decimals: Optional[int]) -> bool:
    if claim.qualifier == "approx":
        return abs(o - c) <= max(abs(c) * 0.05, 0.5 * 10 ** (-claim.decimals))
    if claim.qualifier == "gt":
        return o >= c - 1e-9
    if claim.qualifier == "lt":
        return o <= c + 1e-9
    decimals = claim.decimals
    # If the evidence was printed with fewer (but non-zero) decimals, compare at that precision.
    if obs_decimals is not None and 1 <= obs_decimals < decimals:
        decimals = obs_decimals
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    if abs(o - c) <= tol:
        return True
    f = 10 ** claim.decimals
    return abs(math.trunc(o * f + (1e-9 if o >= 0 else -1e-9)) / f - c) < 1e-9


def values_match(claim: NumberMention, obs) -> bool:
    return any(_close(c, o, claim, od) for c, o, od in _pairs(claim, obs))


def comparable(claim_kinds: Sequence[str], claim_quals: Sequence[str], obs_kinds: Sequence[str], obs_quals: Sequence[str]) -> bool:
    if not claim_kinds or not obs_kinds or not set(claim_kinds) & set(obs_kinds):
        return False
    cq, oq = set(claim_quals) - {"baseline"}, set(obs_quals) - {"baseline"}
    if cq and oq and not cq & oq:
        return False
    if ("baseline" in claim_quals) != ("baseline" in obs_quals):
        return False
    return True


def series_direction(values: Sequence[float]) -> Tuple[Optional[str], bool]:
    """Return (direction, monotonic). Small bumps (<2% of range) are tolerated."""
    if len(values) < 2:
        return None, False
    rng = max(values) - min(values)
    if rng <= 1e-12:
        return "stable", True
    tol = rng * 0.02
    diffs = [b - a for a, b in zip(values, values[1:])]
    ups = sum(1 for d in diffs if d > tol)
    downs = sum(1 for d in diffs if d < -tol)
    overall = values[-1] - values[0]
    if abs(overall) <= rng * 0.05:
        return ("stable", ups == 0 and downs == 0) if ups + downs == 0 else ("mixed", False)
    direction = "increase" if overall > 0 else "decrease"
    monotonic = downs == 0 if direction == "increase" else ups == 0
    return direction, monotonic


def fmt_value(v: float) -> str:
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e12:
        return f"{int(round(v)):,}"
    return f"{v:.4g}"
