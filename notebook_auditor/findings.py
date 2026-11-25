"""Findings mode: draft what a notebook's outputs show, then rate how far to trust each finding.

Offline (heuristic) findings are built directly from labelled printed values, so their numbers are
exact by construction. Claude-drafted findings (including readings of figures) are validated: numbers
must be printed in the cited cell, quotes must be verbatim, and each finding then goes through the same
Planner -> Executor -> Verifier checks as a slide claim.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .auditors.claim_strength_auditor import does_not_prove
from .auditors.figure_auditor import FigureAuditor
from .auditors.risk_auditor import majority_rate
from .claim_extractor import ClaimExtractor
from .code_analysis import CodeIndex
from .models import Claim, ClaimVerdict, EvidenceItem, Finding, Gap, Notebook, Observation, RiskFinding
from .rules import LEAK_CHECKS, related_risks
from .textnum import EVAL_KINDS, extract_numbers, kinds_of_text, series_direction, tokenize, values_match

KIND_LABEL = {
    "accuracy": "accuracy", "f1": "F1", "precision": "precision", "recall": "recall", "auc": "AUC", "loss": "loss",
    "rmse": "RMSE", "mse": "MSE", "mae": "MAE", "r2": "R²", "bleu": "BLEU", "rouge": "ROUGE", "p_value": "p-value",
    "silhouette": "silhouette score", "perplexity": "perplexity", "correlation": "correlation",
}
LOWER_IS_BETTER = {"loss", "rmse", "mse", "mae", "perplexity", "p_value"}
FINDING_KINDS = EVAL_KINDS | {"p_value", "silhouette", "perplexity", "correlation"}
QUAL_LABEL = {"test": "test", "val": "validation", "train": "training", "macro": "macro", "micro": "micro", "weighted": "weighted", "baseline": "baseline"}
SUMMARY_ROW_RE = re.compile(r"avg|accuracy|overall|total|mean", re.I)
NOT_A_METRIC = {"weight", "weights", "rate", "lr", "coef", "coefficient", "penalty", "lambda", "alpha", "threshold", "seed"}
MAX_LLM_IMAGES = 6
MAX_IMAGE_B64 = 3_500_000


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _metric_name(kind: str, quals: Sequence[str]) -> str:
    q = " ".join(QUAL_LABEL[x] for x in quals if x in QUAL_LABEL)
    return f"{q} {KIND_LABEL.get(kind, kind)}".strip()


def _ev(o: Observation, relation: str = "match") -> EvidenceItem:
    return EvidenceItem(o.cell_index, o.location, o.line, o.value, o.label, relation)


def _looks_like_code(label: str) -> bool:
    return bool(re.search(r"[()\[\]{}=]|\bprint\b", label))


def _fmt(v: float, raw: Optional[str] = None) -> str:
    return raw if raw else f"{v:.4g}"


def _norm(s: str) -> str:
    return " ".join(s.split())


class FindingsBuilder:
    def __init__(self, nb: Notebook, index: CodeIndex, observations: Sequence[Observation], figures: FigureAuditor, risks: Sequence[RiskFinding], max_findings: int = 16) -> None:
        self.nb = nb
        self.idx = index
        self.obs = [o for o in observations if o.location == "output"]
        self.figures = figures
        self.risks = list(risks)
        self.max_findings = max_findings
        self.warnings: List[str] = []

    # ------------------------------------------------------------------ offline findings
    def build(self) -> List[Finding]:
        out: List[Finding] = []
        out += self._data()
        series, series_keys = self._series()
        out += self._metrics(skip=series_keys)
        out += series
        out += self._comparisons()
        out += self._bars()
        out += self._significance()
        return out[: self.max_findings]

    def _data(self) -> List[Finding]:
        out: List[Finding] = []
        seen: Set[Tuple[int, str]] = set()
        for o in self.obs:
            if "count" not in o.kinds or (o.cell_index, o.line) in seen:
                continue
            if not re.search(r"load|read|rows|documents|docs|samples|records|examples|dataset|total|shape|\bn\b|sentences|tweets|reviews|texts", f"{o.label} {o.line}", re.I):
                continue
            seen.add((o.cell_index, o.line))
            if o.label.endswith(" rows"):
                cols = next((x for x in self.obs if x.cell_index == o.cell_index and x.line == o.line and x.label.endswith(" columns")), None)
                text = f"The data has {o.raw} rows" + (f" and {cols.raw} columns" if cols else "")
                ev = [_ev(o)] + ([_ev(cols)] if cols else [])
            else:
                text = f"Data size: {o.line.rstrip('.')}"
                ev = [_ev(o)]
            out.append(Finding("", text, "data", "MEASURED", [o.cell_index], ev))
            if len(out) >= 2:
                break
        split_rx = re.compile(r"\b(train\w*|test\w*|val\w*)\b[^\n]{0,25}\b(size|samples|set|rows|examples|n)\b", re.I)
        for cell in self.nb.code_cells:
            line = next((l for l in cell.output_text.splitlines() if split_rx.search(l) and extract_numbers(l) and len(l) < 120), None)
            if line:
                ev = [EvidenceItem(cell.cell_index, "output", line.strip(), None, "split sizes", "match")]
                out.append(Finding("", f"Split sizes: {line.strip()}", "data", "MEASURED", [cell.cell_index], ev))
                break
        mr = majority_rate(self.nb)
        if mr:
            rate, cell_i = mr
            cell = self.nb.get(cell_i)
            lines = [l.strip() for l in (cell.output_text.splitlines() if cell else []) if l.strip()][:4]
            classification = any(set(o.kinds) & {"accuracy", "f1", "precision", "recall"} for o in self.obs)
            phrase = f"The largest class makes up {rate * 100:.1f}% of the labels" if classification else f"The most common category makes up {rate * 100:.1f}% of rows"
            out.append(
                Finding("", phrase, "data", "DERIVED", [cell_i],
                        [EvidenceItem(cell_i, "output", " / ".join(lines), rate, "class counts", "derived")],
                        caveats=[f"A model that always predicts the largest class already scores {rate:.3f} accuracy; judge every accuracy against that."])
            )
        return out

    def _metrics(self, skip: Set[Tuple]) -> List[Finding]:
        groups: Dict[Tuple, Observation] = {}
        for o in self.obs:
            kinds = [k for k in o.kinds if k in FINDING_KINDS and k != "p_value"]
            if not kinds or "support" in o.label.lower():
                continue
            if o.row_key and not SUMMARY_ROW_RE.search(o.row_key):
                continue  # per-row values are handled as comparisons / series
            key = (o.cell_index, kinds[0], tuple(o.qualifiers))
            if key in skip:
                continue  # part of a printed series such as a per-epoch log; reported as a trend instead
            if set(tokenize(o.label)) & NOT_A_METRIC:
                continue  # "class weight for the loss: 9.2" is a setting, not a result
            groups[key] = o  # the last printed value is the final one

        def priority(item):
            (cell, kind, quals), o = item
            return (0 if set(quals) & {"test", "val"} else 2 if "train" in quals else 1, cell)

        out = []
        for (cell, kind, quals), o in sorted(groups.items(), key=priority)[:8]:
            lab = o.label.strip(" :=-|")
            if lab and kinds_of_text(lab) and not _looks_like_code(lab) and len(lab) <= 40:
                text = f"{_cap(lab)} is {o.raw}"
            else:
                text = f"{_cap(_metric_name(kind, quals))} is {o.raw}"
            out.append(Finding("", text, "metric", "MEASURED", [cell], [_ev(o)]))
        return sorted(out, key=lambda f: f.cells[0])

    def _series_finding(self, cell: int, name: str, kind: Optional[str], vals: List[float], raws: List[str], where: List[str], evidence: List[EvidenceItem], plotted: bool) -> Optional[Finding]:
        n = len(vals)
        d, mono = series_direction(vals)
        if d is None:
            return None
        lower = kind in LOWER_IS_BETTER
        best_i = min(range(n), key=lambda i: vals[i]) if lower else max(range(n), key=lambda i: vals[i])
        loc = lambda i: where[i] if where and where[i] else f"point {i + 1} of {n}"
        r = lambda i: raws[i] if raws else f"{vals[i]:.4g}"
        verb = {"increase": "rises", "decrease": "falls", "stable": "stays flat", "mixed": "goes up and down"}[d]
        text = f"{_cap(name)} {verb} from {r(0)} to {r(n - 1)}"
        if d in ("increase", "decrease") and not mono:
            text += ", but not steadily"
        if best_i not in (0, n - 1) or d == "mixed":
            text += f"; best {r(best_i)} at {loc(best_i)}"
        caveats = []
        if plotted:
            caveats.append(f"These values are typed into the plotting code in Cell {cell}, not printed by a computation; check they match the real run.")
        return Finding("", text, "trend", "CODE" if plotted else "DERIVED", [cell], evidence, caveats=caveats)

    def _series(self) -> Tuple[List[Finding], Set[Tuple]]:
        # Table rows and free-standing lines are kept apart: "best depth: 8 (test accuracy 0.8227)"
        # printed under a depth-sweep table is a summary, not the sweep's last point.
        groups: Dict[Tuple[int, str, Tuple[str, ...], bool], List[Observation]] = {}
        for o in self.obs:
            ks = [k for k in o.kinds if k in FINDING_KINDS]
            if not ks:
                continue
            groups.setdefault((o.cell_index, ks[0], tuple(o.qualifiers), bool(o.row_key)), []).append(o)
        out: List[Finding] = []
        keys: Set[Tuple] = set()
        by_cell: Dict[int, Dict[Tuple[str, Tuple[str, ...]], List[Observation]]] = {}
        for (cell, kind, quals, in_table), obs in groups.items():
            if len(obs) < 3 or any(o.row_key and " = " not in o.row_key for o in obs):
                continue  # rows that are classes/models are comparisons, not a series
            obs = sorted(obs, key=lambda o: o.order)
            if not in_table:
                keys.add((cell, kind, quals))  # free-standing lines: don't also report each one as a metric
            by_cell.setdefault(cell, {})[(kind, quals)] = obs
            ev = [_ev(obs[0], "derived"), _ev(obs[-1], "derived")]
            f = self._series_finding(cell, _metric_name(kind, quals), kind, [o.value for o in obs], [o.raw for o in obs], [o.row_key for o in obs], ev, False)
            if f:
                out.append(f)
        for cell, series in by_cell.items():
            for (kind, quals), train in series.items():
                if "train" not in quals:
                    continue
                for (k2, q2), held in series.items():
                    if k2 == kind and set(q2) & {"test", "val"} and len(held) == len(train):
                        f = self._overfitting(cell, kind, train, held, q2)
                        if f:
                            out.append(f)
        for s in self.figures.series:
            kind = next(iter(kinds_of_text(s.label) & FINDING_KINDS), None)
            ev = [EvidenceItem(s.cell_index, "code", f"{s.name or s.label}: {s.values}", None, s.label, "derived")]
            f = self._series_finding(s.cell_index, s.name or s.label, kind, s.values, [], [], ev, True)
            if f:
                out.append(f)
        return out, keys

    def _overfitting(self, cell: int, kind: str, train: List[Observation], held: List[Observation], held_quals) -> Optional[Finding]:
        tv = [o.value for o in train]
        hv = [o.value for o in held]
        n = len(hv)
        lower = kind in LOWER_IS_BETTER
        b = min(range(n), key=lambda i: hv[i]) if lower else max(range(n), key=lambda i: hv[i])
        spread = (max(hv) - min(hv)) or 1.0
        worse_after = (hv[-1] - hv[b]) if lower else (hv[b] - hv[-1])
        train_improves = (tv[-1] < tv[b]) if lower else (tv[-1] > tv[b])
        if b >= n - 1 or worse_after < max(0.01 * abs(hv[b]), 0.1 * spread) or not train_improves:
            return None
        where = held[b].row_key or f"point {b + 1} of {n}"
        held_name = _metric_name(kind, held_quals)
        text = (f"Overfitting: {_metric_name(kind, ['train'])} keeps improving to {train[-1].raw}, while {held_name} "
                f"is best at {held[b].raw} ({where}) and then worsens to {held[-1].raw}")
        ev = [_ev(train[-1], "derived"), _ev(held[b], "derived"), _ev(held[-1], "derived")]
        caveat = ("If the best setting was chosen by looking at test scores, that test score is optimistic; "
                  "choose it on a validation split and report test once.")
        return Finding("", text, "overfitting", "DERIVED", [cell], ev, caveats=[caveat])

    def _comparisons(self) -> List[Finding]:
        groups: Dict[Tuple[int, str], List[Observation]] = {}
        for o in self.obs:
            if not o.row_key or " = " in o.row_key or SUMMARY_ROW_RE.search(o.row_key):
                continue
            if not set(o.kinds) & FINDING_KINDS or "support" in o.label.lower():
                continue
            col = o.label[len(o.row_key):].strip() if o.label.startswith(o.row_key) else o.label
            groups.setdefault((o.cell_index, col), []).append(o)
        out = []
        for (cell, col), obs in groups.items():
            rows = {o.row_key: o for o in obs}
            if len(rows) < 2:
                continue
            kind = next(iter(set(obs[0].kinds) & FINDING_KINDS))
            ordered = sorted(rows.values(), key=lambda o: o.value, reverse=kind not in LOWER_IS_BETTER)
            best, worst = ordered[0], ordered[-1]
            good, bad = ("best", "worst") if kind in EVAL_KINDS else ("highest", "lowest")
            text = f"{_cap(col or KIND_LABEL.get(kind, kind))}: {good} for {best.row_key} ({best.raw}), {bad} for {worst.row_key} ({worst.raw})"
            gap = abs(best.value - worst.value)
            caveats = []
            if len(ordered) >= 2 and abs(ordered[0].value - ordered[1].value) < 0.02 * max(abs(ordered[0].value), 1e-9) + 1e-9:
                caveats.append(f"{ordered[0].row_key} and {ordered[1].row_key} differ by only {abs(ordered[0].value - ordered[1].value):.3g}; one run cannot separate them.")
            out.append(Finding("", text, "comparison", "MEASURED", [cell], [_ev(best), _ev(worst)], caveats=caveats))
            if len(out) >= 4:
                break
        return out

    def _bars(self) -> List[Finding]:
        out = []
        for b in self.figures.bars:
            kinds = kinds_of_text(b.label) & FINDING_KINDS
            lower = bool(kinds & LOWER_IS_BETTER)
            pairs = sorted(zip(b.categories, b.values), key=lambda p: p[1], reverse=not lower)
            (bc, bv), (wc, wv) = pairs[0], pairs[-1]
            text = f"{_cap(b.label)} (bar chart): highest for {bc} ({bv:g}), lowest for {wc} ({wv:g})" if not lower else f"{_cap(b.label)} (bar chart): best (lowest) for {bc} ({bv:g}), worst for {wc} ({wv:g})"
            caveats = [f"These bar heights are typed into the plotting code in Cell {b.cell_index}, not computed there; check they match the real results."]
            if len(pairs) > 1 and abs(pairs[0][1] - pairs[1][1]) < 0.02:
                caveats.append(f"{pairs[0][0]} and {pairs[1][0]} differ by only {abs(pairs[0][1] - pairs[1][1]):.3g}; without repeated runs that gap may be noise.")
            ev = [EvidenceItem(b.cell_index, "code", f"{b.categories} -> {b.values}", None, b.label, "derived")]
            out.append(Finding("", text, "comparison", "CODE", [b.cell_index], ev, caveats=caveats))
        return out

    def _significance(self) -> List[Finding]:
        out = []
        seen: Set[Tuple[int, str]] = set()
        for o in self.obs:
            if "p_value" not in o.kinds or (o.cell_index, o.line) in seen:
                continue
            seen.add((o.cell_index, o.line))
            v = o.value / 100 if o.is_percent else o.value
            side = "below" if v < 0.05 else "not below"
            out.append(Finding("", f"Statistical test: {o.line.rstrip('.')}, {side} the usual 0.05 cutoff", "significance", "MEASURED", [o.cell_index], [_ev(o)]))
        return out[:3]

    # ------------------------------------------------------------------ trust
    def rate(self, f: Finding) -> None:
        pseudo = ClaimExtractor._make_claim(f.text, f.text, "finding", "") or Claim("", f.text, f.text, "finding", "", "qualitative")
        pseudo.claim_id = f.finding_id
        rel = related_risks(pseudo, self.risks, f.cells)
        rel += [r for r in self.risks if r.check == "borderline_threshold" and set(r.cell_indices) & set(f.cells) and r not in rel]
        if f.kind == "data":
            rel = [r for r in rel if r.check not in LEAK_CHECKS]
        blocking = [r for r in rel if (r.check in LEAK_CHECKS and r.severity == "high") or r.check in ("near_chance", "borderline_threshold")]
        other = [r for r in rel if r not in blocking and r.severity in ("high", "medium")]
        f.risk_ids = sorted({r.finding_id for r in rel})
        if f.trust == "solid" and blocking:
            f.trust = "at_risk"
        caveats = [r.message for r in blocking[:2]] + [r.message for r in other[:1]] + f.caveats
        dnp = does_not_prove(pseudo)
        if dnp and f.kind != "data" and not dnp.startswith("This interpretation goes beyond"):  # skip the generic fallback
            caveats.append(dnp)
        seen: Set[str] = set()
        f.caveats = [c for c in caveats if not (c in seen or seen.add(c))][:4]

    # ------------------------------------------------------------------ Claude-drafted findings
    def llm_candidates(self, llm, existing: Sequence[Finding]) -> List[Tuple[Finding, Claim]]:
        from .prompts import FINDINGS_SYSTEM, findings_user_prompt

        text = self.nb.to_linear_text(max_output_chars=1500, include_markdown=True)
        if len(text) > 30000:
            text = text[:30000] + "\n... [notebook truncated]"
        images: List[Tuple[str, str, str]] = []
        for cell in self.nb.code_cells:
            for img in cell.images:
                if img.data and img.mime in ("image/png", "image/jpeg") and len(img.data) <= MAX_IMAGE_B64 and len(images) < MAX_LLM_IMAGES:
                    images.append((f"Figure from Cell {cell.cell_index}:", img.mime, img.data))
        data = llm.complete_json(FINDINGS_SYSTEM, findings_user_prompt(text, len(images)), max_tokens=3000, images=images or None)
        if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
            self.warnings.append("Claude did not return findings; showing offline findings only.")
            return []
        pairs: List[Tuple[Finding, Claim]] = []
        for item in data["findings"][:12]:
            try:
                ci = int(item.get("cell_index"))
                ftext = str(item.get("text") or "").strip()
            except (TypeError, ValueError, AttributeError):
                continue
            cell = self.nb.get(ci)
            if cell is None or cell.cell_type != "code" or not ftext:
                continue
            kind = str(item.get("kind") or "metric").lower()
            if kind not in ("metric", "trend", "comparison", "data", "figure", "interpretation"):
                kind = "interpretation"
            quote = _norm(str(item.get("quote") or ""))
            if quote and quote not in _norm(cell.output_text + "\n" + cell.source_code):
                self.warnings.append(f"Dropped a Claude finding whose quote is not in Cell {ci}: {ftext[:80]!r}")
                continue
            nums = extract_numbers(ftext, exclude_refs=True)
            cell_obs = [o for o in self.obs if o.cell_index == ci]
            if kind != "figure" and any(not any(values_match(n, o) for o in cell_obs) for n in nums):
                self.warnings.append(f"Dropped a Claude finding with numbers not printed in Cell {ci}: {ftext[:80]!r}")
                continue
            if any(ci in f.cells and any(e.value is not None and any(values_match(n, _as_obs(e)) for n in nums) for e in f.evidence) for f in existing):
                continue  # already covered by an offline finding
            basis = "VISUAL" if kind == "figure" else "INTERPRETATION" if kind == "interpretation" else "MEASURED"
            ev = [EvidenceItem(ci, "output", quote or ("figure" if kind == "figure" else ""), None, "", "match" if quote else "context")]
            finding = Finding("", ftext.rstrip("."), kind, basis, [ci], [e for e in ev if e.snippet], trust="visual" if kind == "figure" else "solid", source="llm")
            claim = ClaimExtractor._make_claim(ftext, ftext, f"claude-finding:Cell {ci}", "", extractor="llm") or Claim("", ftext, ftext, f"claude-finding:Cell {ci}", "", "qualitative", extractor="llm")
            pairs.append((finding, claim))
        return pairs

    def accept_llm(self, f: Finding, v: ClaimVerdict) -> bool:
        if v.verdict == "CONTRADICTED":
            self.warnings.append(f"Dropped a Claude finding that the outputs contradict: {f.text[:80]!r} ({v.rationale[:120]})")
            return False
        if v.verdict == "METHODOLOGICAL_RISK":
            f.trust = "at_risk"
        elif v.verdict == "UNVERIFIABLE":
            f.trust = "visual" if f.kind == "figure" else "unconfirmed"
        f.evidence = f.evidence or v.evidence[:2]
        return True


def _as_obs(e: EvidenceItem) -> Observation:
    raw = f"{e.value}"
    dec = len(raw.split(".")[1]) if "." in raw else 0
    return Observation(e.cell_index, e.location, float(e.value or 0), raw, False, dec, e.label, [], [], e.snippet)


# --------------------------------------------------------------------------- what's missing
PIPELINE_SNIPPET = '''from sklearn.pipeline import make_pipeline
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, stratify=y, random_state=42)
model = make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)   # the vectoriser and scaler now learn from training data only
print("Test accuracy:", model.score(X_test, y_test))'''

BASELINE_SNIPPET = '''from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("Majority-class accuracy:", baseline.score(X_test, y_test))'''

SEEDS_SNIPPET = '''import numpy as np
scores = []
for seed in [0, 1, 2, 3, 4]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    scores.append(make_model().fit(X_tr, y_tr).score(X_te, y_te))   # make_model() builds a fresh model
print(f"accuracy = {np.mean(scores):.3f} ± {np.std(scores):.3f} over {len(scores)} seeds")'''

BOOTSTRAP_SNIPPET = '''import numpy as np
rng = np.random.default_rng(0)
correct = (np.asarray(preds) == np.asarray(y_test)).astype(float)
boot = [rng.choice(correct, len(correct)).mean() for _ in range(1000)]
print("accuracy 95% CI:", np.percentile(boot, [2.5, 97.5]))'''

GROUP_SNIPPET = '''from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["doc_id"]))   # each document stays in one split'''

SPLIT_SNIPPET = '''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))'''

REPORT_SNIPPET = '''from sklearn.metrics import classification_report, ConfusionMatrixDisplay
print(classification_report(y_test, preds, digits=3))
ConfusionMatrixDisplay.from_predictions(y_test, preds)'''


def build_gaps(nb: Notebook, idx: CodeIndex, risks: Sequence[RiskFinding], observations: Sequence[Observation], figures: Optional[FigureAuditor] = None) -> List[Gap]:
    checks = {r.check for r in risks}

    def cells_of(*names: str) -> List[int]:
        return sorted({c for r in risks if r.check in names for c in r.cell_indices})

    outputs = [o for o in observations if o.location == "output" and not set(tokenize(o.label)) & NOT_A_METRIC]
    has_metrics = any(set(o.kinds) & EVAL_KINDS for o in outputs)
    is_classification = any(set(o.kinds) & {"accuracy", "f1", "precision", "recall"} for o in outputs)
    gaps: List[Gap] = []
    if checks & {"data_leakage", "test_set_contamination"}:
        gaps.append(Gap("Fit preprocessing on the training data only",
                        "A scaler, vectoriser or imputer learned from rows that later became test data, so the test score is likely too optimistic. Wrapping the steps in a Pipeline fixes it.",
                        PIPELINE_SNIPPET, "high", cells_of("data_leakage", "test_set_contamination")))
    if "no_holdout" in checks:
        gaps.append(Gap("Evaluate on data the model has not seen", "The metrics appear to be computed on the training data.", SPLIT_SNIPPET, "high", cells_of("no_holdout")))
    if "group_leakage" in checks:
        gaps.append(Gap("Keep each document or user in a single split", "Rows sharing an ID can end up in both train and test, which inflates scores.", GROUP_SNIPPET, "medium", cells_of("group_leakage")))
    if "near_chance" in checks:
        gaps.append(Gap("Check the model beats the majority class", "Accuracy is close to what always predicting the largest class would get.", BASELINE_SNIPPET, "high", cells_of("near_chance")))
    elif "missing_baseline" in checks:
        gaps.append(Gap("Add a baseline", "Without a majority-class or random baseline, a reader cannot tell whether the score is good for this data.", BASELINE_SNIPPET, "medium", cells_of("missing_baseline")))
    if has_metrics and not idx.source_matches(r"cross_val|KFold|for\s+\w*seed\w*\s+in|seeds?\s*=\s*[\[(]|n_repeats|range\(\s*n_runs"):
        gaps.append(Gap("Repeat the experiment with several seeds or folds",
                        "Every number comes from a single split and a single run, so differences of a point or two may be noise.", SEEDS_SNIPPET, "medium"))
    if has_metrics and not idx.source_matches(r"\.std\(|np\.std|confidence|bootstrap|\bsem\b|\bci\b|yerr|errorbar|percentile"):
        gaps.append(Gap("Report uncertainty", "No standard deviation, confidence interval or error bars are reported for the metrics.", BOOTSTRAP_SNIPPET, "medium"))
    if "borderline_threshold" in checks:
        msgs = "; ".join(r.message for r in risks if r.check == "borderline_threshold")
        gaps.append(Gap("Don't rest a conclusion on a borderline number",
                        f"{msgs}.",
                        "Report the effect size with a 95% confidence interval, say how many tests or thresholds you tried (apply Holm–Bonferroni if several), and describe the result as suggestive rather than conclusive.",
                        "medium", cells_of("borderline_threshold")))
    if is_classification and not idx.source_matches(r"classification_report|confusion_matrix|ConfusionMatrixDisplay|average\s*=\s*None"):
        gaps.append(Gap("Look at per-class results", "Only overall scores are printed; a weak class can hide behind a good average, especially with imbalanced labels.", REPORT_SNIPPET, "low"))
    if checks & {"execution_order", "cell_errors", "unexecuted_cells"}:
        gaps.append(Gap("Restart and run all cells before trusting the outputs",
                        "Cells ran out of order, failed, or never ran, so the saved outputs may not match the code.",
                        "Colab: Runtime → Restart session and run all. Jupyter: Kernel → Restart & Run All. Then save and audit again.",
                        "medium", cells_of("execution_order", "cell_errors", "unexecuted_cells")))
    if figures is not None and figures.bars:
        gaps.append(Gap("Plot computed results, not typed-in numbers",
                        "Some charts draw values written by hand into the code, so they can silently drift from the real results.",
                        "results = {\"LogReg\": acc_lr, \"SVM\": acc_svm}   # variables computed earlier\nplt.bar(list(results), list(results.values()))",
                        "low", sorted({b.cell_index for b in figures.bars})))
    if "figure_not_rendered" in checks:
        gaps.append(Gap("Save figures with the notebook", "Some plotting cells have no saved image, so nobody can check them.",
                        "Run the plotting cells, then save. In Colab, check Edit → Notebook settings → 'Omit code cell output when saving' is off.", "low", cells_of("figure_not_rendered")))
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(gaps, key=lambda g: order.get(g.severity, 3))
