"""Methodological risk auditor: data leakage, missing baselines, borderline thresholds, hygiene."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..code_analysis import CallEvent, CodeIndex
from ..models import Claim, Notebook, Observation, RiskFinding
from ..textnum import BOUNDED_KINDS, EVAL_KINDS, extract_numbers, fmt_value, tokenize

TRANSFORMERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "QuantileTransformer", "PowerTransformer",
    "TfidfVectorizer", "CountVectorizer", "TfidfTransformer", "DictVectorizer", "SimpleImputer", "KNNImputer",
    "IterativeImputer", "PCA", "TruncatedSVD", "NMF", "LatentDirichletAllocation", "SelectKBest",
    "SelectPercentile", "RFE", "RFECV", "SelectFromModel", "VarianceThreshold", "OneHotEncoder",
    "OrdinalEncoder", "TargetEncoder", "KBinsDiscretizer", "LabelEncoder", "Word2Vec",
}
RESAMPLERS = {"SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "RandomOverSampler", "RandomUnderSampler", "SMOTETomek", "SMOTEENN"}
SPLIT_FUNCS = {"train_test_split"}
CV_FUNCS = {"cross_val_score", "cross_validate", "cross_val_predict", "learning_curve", "validation_curve"}
SEARCH_CLASSES = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV"}
SPLITTER_CLASSES = {"KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit", "TimeSeriesSplit", "RepeatedKFold", "RepeatedStratifiedKFold", "LeaveOneOut"}
GROUP_SPLITTERS = {"GroupKFold", "GroupShuffleSplit", "StratifiedGroupKFold", "LeaveOneGroupOut", "LeavePGroupsOut"}
FIT_ATTRS = {"fit", "fit_transform", "fit_resample"}
METRIC_FUNCS = {
    "accuracy_score", "f1_score", "precision_score", "recall_score", "roc_auc_score", "classification_report",
    "balanced_accuracy_score", "r2_score", "mean_squared_error", "mean_absolute_error", "matthews_corrcoef",
    "log_loss", "precision_recall_fscore_support",
}
CLASSIFIER_METRIC_FUNCS = {
    "accuracy_score", "f1_score", "precision_score", "recall_score", "roc_auc_score", "classification_report",
    "balanced_accuracy_score", "precision_recall_fscore_support",
}
CLASSIFIER_METRICS = {"accuracy", "f1", "precision", "recall", "auc"}
BASELINE_RE = re.compile(r"DummyClassifier|DummyRegressor|most_frequent|majority|baseline|chance|random[ _]guess|stratified['\"]", re.I)
ID_COLUMN_RE = re.compile(r"\b(doc(?:ument)?_?id|user_?id|patient_?id|subject_?id|author_?id|group_?id|session_?id|speaker_?id|participant_?id|customer_?id|article_?id)\b", re.I)
DEFAULT_ALPHAS = [0.05]


def _f(engine_check: str, severity: str, message: str, cells: Sequence[int], detail: str = "", claim_ids=None) -> RiskFinding:
    return RiskFinding("", "risk", engine_check, severity, message, sorted(set(cells)), detail, list(claim_ids or []))


class RiskAuditor:
    def __init__(self, nb: Notebook, index: CodeIndex, observations: Sequence[Observation], claims: Sequence[Claim], threshold_tolerance: float = 0.005) -> None:
        self.nb = nb
        self.idx = index
        self.obs = list(observations)
        self.claims = list(claims)
        self.tol = threshold_tolerance

    def run(self) -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        findings += self.check_leakage()
        findings += self.check_baseline()
        findings += self.check_borderline_thresholds()
        findings += self.check_hygiene()
        return findings

    # ------------------------------------------------------------------ helpers
    def _split_events(self) -> List[CallEvent]:
        ev = []
        for c in self.idx.calls:
            cls = self.idx.class_of(c)
            if c.attr in SPLIT_FUNCS or c.attr in CV_FUNCS:
                ev.append(c)
            elif c.attr == "split" and cls in SPLITTER_CLASSES | GROUP_SPLITTERS:
                ev.append(c)
            elif c.attr == "fit" and cls in SEARCH_CLASSES:
                ev.append(c)
        return ev

    def _transformer_fits(self) -> List[Tuple[CallEvent, str]]:
        out = []
        for c in self.idx.calls_to(*FIT_ATTRS):
            cls = self.idx.class_of(c)
            if cls in TRANSFORMERS or cls in RESAMPLERS:
                out.append((c, cls))
        return out

    # ------------------------------------------------------------------ leakage
    def check_leakage(self) -> List[RiskFinding]:
        out: List[RiskFinding] = []
        splits = self._split_events()
        fits = self._transformer_fits()
        if splits:
            first = min(splits, key=lambda c: c.order)
            split_inputs: Set[str] = set()
            for s in splits:
                split_inputs |= s.all_arg_names
            for call, cls in fits:
                if call.order >= first.order:
                    continue
                tainted = self.idx.taint_forward(set(call.targets) | call.all_arg_names, call.order)
                flows = tainted & split_inputs
                severity = "high" if flows or cls in RESAMPLERS else "medium"
                what = "resampled (synthetic/duplicated rows)" if cls in RESAMPLERS else "fitted on the full dataset"
                out.append(
                    _f(
                        "data_leakage", severity,
                        f"{cls} is {what} in Cell {call.cell_index} before the train/test split in Cell {first.cell_index}",
                        [call.cell_index, first.cell_index],
                        f"`{call.source_line}` (Cell {call.cell_index}, line {call.lineno}) runs before "
                        f"`{first.source_line}` (Cell {first.cell_index}, line {first.lineno})."
                        + (f" Data flow: {', '.join(sorted(flows))} feeds the split." if flows else "")
                        + " Statistics learned from test rows leak into training; fit the transformer on the training split only (e.g. inside a Pipeline).",
                    )
                )
            # pandas-level whole-dataset statistics before the split
            for cell_i, ln, line in self.idx.source_matches(r"fillna\([^)]*\.(mean|median|mode)\(|\(\s*\w+(\[[^\]]*\])?\s*-\s*\w+(\[[^\]]*\])?\.(mean|min)\(\)\s*\)\s*/"):
                if (cell_i, ln) < (first.cell_index, first.lineno):
                    out.append(
                        _f("data_leakage", "medium",
                           f"Dataset-wide statistic computed in Cell {cell_i} before the split in Cell {first.cell_index}",
                           [cell_i, first.cell_index], f"`{line}` uses the mean/median of all rows, including future test rows.")
                    )
            # fitting on the test partition
            test_names: Set[str] = set()
            for s in splits:
                if s.attr in SPLIT_FUNCS:
                    tg = s.targets
                    if len(tg) >= 2:
                        test_names |= {t for t in tg if "test" in t.lower()} or ({tg[1], tg[3]} if len(tg) == 4 else {tg[1]})
            for c in self.idx.calls_to("fit", "fit_transform", "fit_resample", "partial_fit"):
                if c.arg_names and c.arg_names[0] & test_names and c.order > first.order:
                    out.append(
                        _f("test_set_contamination", "high",
                           f"`{c.func}` is fitted on test data in Cell {c.cell_index}",
                           [c.cell_index], f"`{c.source_line}` — test partition {sorted(c.arg_names[0] & test_names)} must only be passed to transform/predict.")
                    )
            # group / document-ID leakage
            id_hits = self.idx.source_matches(ID_COLUMN_RE.pattern)
            uses_group_split = any(self.idx.class_of(c) in GROUP_SPLITTERS or c.attr in GROUP_SPLITTERS for c in self.idx.calls) or any("groups" in c.kwargs for c in splits)
            if id_hits and not uses_group_split:
                cells = sorted({h[0] for h in id_hits})
                ids = sorted({m.group(0) for h in id_hits for m in ID_COLUMN_RE.finditer(h[2])})
                out.append(
                    _f("group_leakage", "medium",
                       f"Identifier column(s) {', '.join(ids)} present but the split is not group-aware",
                       cells + [first.cell_index],
                       "Rows sharing an ID (same document/user) can land in both train and test. Use GroupShuffleSplit/GroupKFold or verify the ID intersection is empty.")
                )
            out += self._overlap_outputs()
        else:
            evals = self.idx.calls_to(*METRIC_FUNCS) + [c for c in self.idx.calls_to("score") if c.receiver]
            fits_models = [c for c in self.idx.calls_to("fit") if self.idx.class_of(c) not in TRANSFORMERS]
            if evals and fits_models:
                out.append(
                    _f("no_holdout", "high",
                       "Metrics are computed but no train/test split or cross-validation was found",
                       [evals[0].cell_index, fits_models[0].cell_index],
                       f"`{fits_models[0].source_line}` then `{evals[0].source_line}` — the model is likely evaluated on its own training data.")
                )
        return out

    def _overlap_outputs(self) -> List[RiskFinding]:
        out = []
        for cell in self.nb.code_cells:
            src = cell.source_code
            if not (ID_COLUMN_RE.search(src) or "id" in src.lower()) or not re.search(r"intersection|&|isin\(", src):
                continue
            for o in self.obs:
                if o.cell_index == cell.cell_index and o.location == "output" and re.search(r"overlap|shared|common|intersect|leak", f"{o.label} {o.line}", re.I):
                    if o.value > 0:
                        out.append(_f("group_leakage", "high", f"Cell {cell.cell_index} reports {fmt_value(o.value)} IDs shared between train and test", [cell.cell_index], f"Output: `{o.line}`"))
        return out

    # ------------------------------------------------------------------ baseline
    def check_baseline(self) -> List[RiskFinding]:
        metric_obs = [o for o in self.obs if o.location == "output" and set(o.kinds) & CLASSIFIER_METRICS and "baseline" not in o.qualifiers]
        metric_claims = [c for c in self.claims if any(set(n.kinds) & CLASSIFIER_METRICS for n in c.numbers)]
        metric_calls = self.idx.calls_to(*CLASSIFIER_METRIC_FUNCS)  # regression metrics have no "chance level"
        if not (metric_obs or metric_claims or metric_calls):
            return []
        code_hits = self.idx.source_matches(BASELINE_RE.pattern)
        out_hits = [o for o in self.obs if "baseline" in o.qualifiers]
        out: List[RiskFinding] = []
        majority = self._majority_rate()
        if not code_hits and not out_hits:
            cells = sorted({o.cell_index for o in metric_obs} | {c.cell_index for c in metric_calls})
            detail = "Report a DummyClassifier(strategy='most_frequent') / chance-level score next to every accuracy or F1."
            if majority:
                detail += f" Class distribution in Cell {majority[1]} implies a majority-class accuracy of {majority[0]:.3f}."
            out.append(
                _f("missing_baseline", "medium",
                   "Accuracy/F1 is reported without a random, majority-class or chance-level baseline",
                   cells, detail, [c.claim_id for c in metric_claims])
            )
        if majority:
            rate, cell_i = majority
            near = []
            for o in metric_obs:
                if "accuracy" not in o.kinds or "train" in o.qualifiers:
                    continue
                if o.is_percent:
                    acc = o.value / 100
                elif 0 < o.value <= 1 and o.decimals >= 1:
                    acc = o.value
                else:
                    continue  # integers / out-of-range values are not accuracies
                if acc <= rate + 0.02:
                    near.append(o)
            if near:
                out.append(
                    _f("near_chance", "high",
                       f"Accuracy {', '.join(o.raw for o in near[:4])} is within 2 points of the majority-class rate {rate:.3f} (Cell {cell_i})",
                       [o.cell_index for o in near] + [cell_i], "The classifier may be doing little more than predicting the most frequent class.")
                )
        return out

    def _majority_rate(self) -> Optional[Tuple[float, int]]:
        return majority_rate(self.nb)

    # ------------------------------------------------------------------ thresholds
    def _alphas(self) -> List[Tuple[float, str, Optional[int]]]:
        found: List[Tuple[float, str, Optional[int]]] = []
        for cell_i, _ln, name, value in self.idx.numeric_assigns:
            if re.search(r"alpha|signif|sig_level|p_?thresh", name, re.I) and 0 < value < 0.2:
                found.append((value, f"`{name} = {value}`", cell_i))
        for cell_i, _ln, left, _op, value in self.idx.compares:
            if re.search(r"\bp(_?val(ue)?)?\b|pvalue", left, re.I) and 0 < value < 0.2:
                found.append((value, f"`{left}` compared to {value}", cell_i))
        for c in self.claims:
            for m in re.finditer(r"(?:α|alpha|significance level)\s*(?:of|=|:)?\s*(0?\.\d+)", c.text, re.I):
                found.append((float(m.group(1)), f"{c.claim_id} text", None))
        if not found:
            found = [(a, "conventional α", None) for a in DEFAULT_ALPHAS]
        seen, uniq = set(), []
        for a in found:
            if a[0] not in seen:
                seen.add(a[0])
                uniq.append(a)
        return uniq

    def _metric_thresholds(self) -> List[Tuple[float, str, Optional[int], Set[str]]]:
        found = []
        for cell_i, _ln, name, value in self.idx.numeric_assigns:
            if re.search(r"thresh|cutoff|target|goal|min_|required|pass", name, re.I) and not re.search(r"alpha|signif", name, re.I):
                found.append((value, f"`{name} = {value}`", cell_i, set(tokenize(name))))
        for cell_i, _ln, left, _op, value in self.idx.compares:
            ltoks = set(tokenize(left))
            if re.search(r"\bp(_?val(ue)?)?\b|pvalue|predict_proba|proba", left, re.I):
                continue
            if ltoks & {"acc", "accuracy", "f1", "score", "precision", "recall", "auc", "r2", "metric"}:
                found.append((value, f"`{left}` compared to {value}", cell_i, ltoks))
        for c in self.claims:
            for m in re.finditer(r"(?:threshold|target|cut-?off|bar|requirement|goal)\s*(?:of|=|:)?\s*(\d*\.\d+|\d+%?)", c.text, re.I):
                raw = m.group(1)
                val = float(raw.rstrip("%")) / (100 if raw.endswith("%") else 1)
                found.append((val, f"{c.claim_id} text", None, set(tokenize(c.text))))
        return found

    def check_borderline_thresholds(self) -> List[RiskFinding]:
        out: List[RiskFinding] = []
        tol = self.tol + 1e-9
        values: List[Tuple[float, str, Optional[int], List[str], Optional[str]]] = []  # value, raw, cell, kinds, claim_id
        for o in self.obs:
            if o.location == "output" and o.kinds:
                v = o.value / 100 if o.is_percent else o.value
                values.append((v, o.raw, o.cell_index, o.kinds, None))
        for c in self.claims:
            for n in c.numbers:
                near = f"{n.before[-25:]} {n.after[:25]}"
                if n.kinds and "threshold" not in n.kinds and not re.search(r"threshold|target|cut-?off|alpha|α|goal", near, re.I):
                    v = n.value / 100 if n.is_percent else n.value
                    values.append((v, n.raw, None, n.kinds, c.claim_id))
        # One finding per (value, threshold); notebook outputs and slide claims about it are merged.
        merged: Dict[Tuple, RiskFinding] = {}

        def add(key, cell, cid, factory) -> None:
            f = merged.get(key)
            if f is None:
                merged[key] = f = factory()
            if cell:
                f.cell_indices = sorted(set(f.cell_indices) | {cell})
            if cid and cid not in f.claim_ids:
                f.claim_ids.append(cid)
                f.severity = "high"  # a derived artifact draws a conclusion from the borderline value

        for alpha, alpha_src, alpha_cell in self._alphas():
            for v, raw, cell, kinds, cid in values:
                if "p_value" in kinds and abs(v - alpha) <= tol:
                    add(("p", alpha), cell, cid, lambda raw=raw, v=v, alpha=alpha, alpha_src=alpha_src, alpha_cell=alpha_cell: _f(
                        "borderline_threshold", "medium",
                        f"p = {raw} is within {self.tol} of α = {alpha}",
                        [alpha_cell] if alpha_cell else [],
                        f"Significance decision hinges on a {abs(v - alpha):.4f} margin (α from {alpha_src}). "
                        "Report effect sizes/CIs and check robustness (multiple-comparison correction, other seeds, bootstrap)."))
        for thr, thr_src, thr_cell, _toks in self._metric_thresholds():
            for v, raw, cell, kinds, cid in values:
                if "p_value" in kinds or not (set(kinds) & (EVAL_KINDS | BOUNDED_KINDS)):
                    continue
                if abs(v - thr) <= tol:
                    add(("m", thr, tuple(kinds)), cell, cid, lambda raw=raw, v=v, thr=thr, thr_src=thr_src, thr_cell=thr_cell, kinds=kinds: _f(
                        "borderline_threshold", "medium",
                        f"{'/'.join(kinds)} = {raw} is within {self.tol} of the threshold {thr:g}",
                        [thr_cell] if thr_cell else [],
                        f"Threshold from {thr_src}. A margin of {abs(v - thr):.4f} is within typical run-to-run noise; "
                        "report variance across seeds/folds before claiming the threshold is met."))
        out.extend(merged.values())
        return out

    # ------------------------------------------------------------------ hygiene
    def check_hygiene(self) -> List[RiskFinding]:
        out: List[RiskFinding] = []
        code = self.nb.code_cells
        counts = [(c.cell_index, c.execution_count) for c in code if c.execution_count is not None]
        if counts:
            seq = [n for _, n in counts]
            if any(b <= a for a, b in zip(seq, seq[1:])):
                bad = [counts[i + 1][0] for i in range(len(seq) - 1) if seq[i + 1] <= seq[i]]
                out.append(
                    _f("execution_order", "medium", "Cells were executed out of top-to-bottom order", bad,
                       f"Execution counts in cell order: {seq}. Saved outputs may not be reproducible by 'Run All'.")
                )
            unexecuted = [c.cell_index for c in code if c.execution_count is None and c.source_code.strip()]
            if unexecuted:
                out.append(_f("unexecuted_cells", "low", "Some code cells have no execution count (never run in the saved state)", unexecuted))
        errors = [c.cell_index for c in code if c.has_error]
        if errors:
            out.append(_f("cell_errors", "medium", "Cells raised errors; later outputs may come from a stale kernel state", errors))
        seeds = self.idx.source_matches(r"random_state\s*=\s*(?!None)|np\.random\.seed|random\.seed|torch\.manual_seed|set_seed|seed_everything")
        stochastic = self.idx.source_matches(r"train_test_split|KFold|RandomForest|TSNE|KMeans|shuffle|\.sample\(|Dropout|SGD")
        if stochastic and not seeds:
            out.append(_f("no_random_seed", "low", "Stochastic steps found but no random seed is fixed", sorted({h[0] for h in stochastic})))
        if self.idx.parse_errors:
            out.append(_f("unparsed_cells", "info", "Some code cells could not be parsed; leakage checks for them used regex only", self.idx.parse_errors))
        return out


def majority_rate(nb: Notebook) -> Optional[Tuple[float, int]]:
    """Majority-class share from a printed value_counts() of a label-like column."""
    label_like = re.compile(r"(label|target|class|sentiment|category|y)\b[^\n]*value_counts|y\w*\.value_counts", re.I)
    for cell in nb.code_cells:
        if not label_like.search(cell.source_code):
            continue
        for out in cell.outputs:
            block: List[float] = []
            for line in out.text.splitlines() + [""]:
                parts = line.rsplit(None, 1)  # "positive    252" -> ["positive", "252"]; linear, no regex backtracking
                head = parts[0].strip() if len(parts) == 2 else ""
                if (head and not head[0].isdigit() and ":" not in head and re.fullmatch(r"\d+(?:\.\d+)?", parts[1])
                        and not re.match(r"(Name|dtype|Length)\b", head)):
                    block.append(float(parts[1]))
                    continue
                if len(block) >= 2 and sum(block) > 0:
                    return max(block) / sum(block), cell.cell_index
                block = []
    return None
