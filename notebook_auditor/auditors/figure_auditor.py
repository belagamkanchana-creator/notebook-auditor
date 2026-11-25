"""Visual & figure auditor: axis labels, metric bounds, rendering, and plotted-trend consistency."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set

from ..code_analysis import CallEvent, CodeIndex, literal_list, literal_number
from ..models import Claim, Notebook, RiskFinding, SeriesEvidence
from ..textnum import BOUNDED_KINDS, content_tokens, kinds_of_text, series_direction

PLOT_ATTRS = {
    "plot", "bar", "barh", "scatter", "hist", "boxplot", "violinplot", "errorbar", "fill_between", "imshow", "pie",
    "step", "stem", "semilogy", "semilogx", "loglog", "matshow", "heatmap", "lineplot", "barplot", "scatterplot",
    "histplot", "kdeplot", "countplot", "catplot", "relplot", "lmplot", "pairplot", "displot", "pointplot",
    "stripplot", "swarmplot", "regplot", "plot_confusion_matrix", "from_predictions", "from_estimator",
}
PLOT_ROOTS = {"plt", "pyplot", "sns", "ax", "axes", "axs", "ax1", "ax2", "ax3", "fig", "px", "go"}
NO_AXIS_LABEL_NEEDED = {"imshow", "pie", "heatmap", "matshow", "pairplot", "plot_confusion_matrix", "from_predictions", "from_estimator"}
BAR_ATTRS = {"bar", "barh", "barplot", "countplot"}
LINE_ATTRS = {"plot", "lineplot", "semilogy", "semilogx", "loglog", "step", "errorbar"}
DIMRED_RE = re.compile(r"\b(TSNE|UMAP|umap\.UMAP|PCA|Isomap|MDS)\b")
FIGURE_REF_RE = re.compile(r"\b(figure|fig\.?|plot|chart|graph|visuali[sz]ation|t-?sne|umap|heatmap|curve|scatter)\b", re.I)


def _is_plot_call(c: CallEvent) -> bool:
    if c.attr not in PLOT_ATTRS:
        return False
    root = (c.receiver or "").split(".")[0].split("[")[0].rstrip("()")
    return root in PLOT_ROOTS or (c.receiver or "").endswith(".plot") or c.attr == "plot" or root.startswith("ax")


def _f(check: str, severity: str, message: str, cells: Sequence[int], detail: str = "", claim_ids=None) -> RiskFinding:
    return RiskFinding("", "figure", check, severity, message, sorted(set(cells)), detail, list(claim_ids or []))


class FigureAuditor:
    def __init__(self, nb: Notebook, index: CodeIndex, claims: Sequence[Claim]) -> None:
        self.nb = nb
        self.idx = index
        self.claims = list(claims)
        self.series: List[SeriesEvidence] = []
        self.plot_cells: Dict[int, List[CallEvent]] = {}
        for c in index.calls:
            if _is_plot_call(c):
                self.plot_cells.setdefault(c.cell_index, []).append(c)

    def run(self) -> List[RiskFinding]:
        out: List[RiskFinding] = []
        for cell_i, plots in sorted(self.plot_cells.items()):
            out += self._audit_cell(cell_i, plots)
        out += self._images_without_code()
        out += self._trend_consistency()
        out += self._dimred_claims()
        return out

    # ------------------------------------------------------------------ per cell
    def _cell_calls(self, cell_i: int) -> List[CallEvent]:
        return [c for c in self.idx.calls if c.cell_index == cell_i]

    def _text_arg(self, calls: List[CallEvent], *attrs: str) -> Optional[str]:
        for c in calls:
            if c.attr in attrs:
                if c.args:
                    return c.args[0].strip("'\"")
                for key in ("label", "xlabel", "ylabel", "t"):
                    if key in c.kwargs:
                        return c.kwargs[key].strip("'\"")
        return None

    def _audit_cell(self, cell_i: int, plots: List[CallEvent]) -> List[RiskFinding]:
        out: List[RiskFinding] = []
        calls = self._cell_calls(cell_i)
        cell = self.nb.get(cell_i)
        attrs = {c.attr for c in calls}
        for c in calls:
            if c.attr == "set" and c.receiver:
                attrs |= {k if k.startswith("set_") else f"set_{k}" for k in c.kwargs}
        kinds_needing = [p for p in plots if p.attr not in NO_AXIS_LABEL_NEEDED]
        sns_labeled = any(p.receiver == "sns" and ("x" in p.kwargs or "y" in p.kwargs) for p in plots)
        pandas_plot = any((p.receiver or "").endswith(".plot") or (p.attr == "plot" and (p.receiver or "").split(".")[0] not in PLOT_ROOTS) for p in plots)
        has_x = bool(attrs & {"xlabel", "set_xlabel", "set_xlabels"}) or sns_labeled or any("xlabel" in p.kwargs for p in plots)
        has_y = bool(attrs & {"ylabel", "set_ylabel", "set_ylabels"}) or sns_labeled or any("ylabel" in p.kwargs for p in plots)
        has_title = bool(attrs & {"title", "set_title", "suptitle"}) or any("title" in p.kwargs for p in plots)
        if kinds_needing and not pandas_plot:
            missing = [n for n, ok in (("x-axis label", has_x), ("y-axis label", has_y)) if not ok]
            if missing:
                out.append(_f("missing_axis_labels", "low", f"Figure in Cell {cell_i} has no {' or '.join(missing)}", [cell_i],
                              "Unlabelled axes make it impossible to check which metric/unit a figure claim refers to."))
        if not has_title and len(plots) > 0 and not pandas_plot:
            out.append(_f("missing_title", "info", f"Figure in Cell {cell_i} has no title", [cell_i]))

        ylabel = self._text_arg(calls, "ylabel", "set_ylabel") or ""
        title = self._text_arg(calls, "title", "set_title", "suptitle") or ""
        metric_kinds = kinds_of_text(f"{ylabel} {title}") | {k for p in plots for k in kinds_of_text(p.kwargs.get("label", ""))}
        # axis bounds
        for c in calls:
            if c.attr in ("ylim", "set_ylim") and c.args:
                bounds = [literal_number(n) for n in c.arg_nodes[:2]] if len(c.arg_nodes) >= 2 else (literal_list(c.arg_nodes[0]) or [])
                lo = bounds[0] if bounds else None
                hi = bounds[1] if len(bounds) > 1 else None
                if lo is not None and lo > 0 and any(p.attr in BAR_ATTRS for p in plots):
                    out.append(_f("truncated_axis", "medium", f"Bar chart in Cell {cell_i} starts its y-axis at {lo:g}, not 0", [cell_i],
                                  f"`{c.source_line}` exaggerates visual differences between bars."))
                if metric_kinds & BOUNDED_KINDS and hi is not None and (hi > 1.0 and hi < 50 or (lo is not None and lo < 0)):
                    out.append(_f("metric_bounds", "low", f"y-limits {bounds} in Cell {cell_i} fall outside the [0, 1] range of {'/'.join(sorted(metric_kinds & BOUNDED_KINDS))}", [cell_i], f"`{c.source_line}`"))
        if "twinx" in attrs or "twiny" in attrs:
            out.append(_f("dual_axes", "low", f"Cell {cell_i} uses twin axes; visual alignment of two scales can imply false correlation", [cell_i]))
        if attrs & {"set_yscale", "yscale", "semilogy", "loglog"}:
            out.append(_f("log_scale", "info", f"Cell {cell_i} uses a log-scaled y-axis; read slopes accordingly", [cell_i]))
        # rendering
        if cell is not None and not cell.images:
            out.append(_f("figure_not_rendered", "medium", f"Plotting code in Cell {cell_i} has no saved image output", [cell_i],
                          "The saved notebook has no rendered figure for this plot, so figure-based claims cannot be checked visually."))
        # literal data -> series
        for p in plots:
            if p.attr not in LINE_ATTRS:  # bars/scatters are not ordered series, so no trend
                continue
            if not p.arg_nodes or p.receiver == "sns":
                continue
            resolved = [self._resolve_list(n) for n in p.arg_nodes]
            values = resolved[1] if len(resolved) >= 2 else resolved[0]  # plot(x, y) or plot(y)
            if not values or len(values) < 2:
                continue
            d, mono = series_direction(values)
            label_kw = p.kwargs.get("label", "").strip("'\"")
            y_src = p.args[1] if len(p.args) >= 2 else p.args[0]
            label = " ".join(x for x in (label_kw, ylabel, title) if x) or y_src
            self.series.append(SeriesEvidence(cell_i, label, values, d, mono, "plot_literal"))
        return out

    def _resolve_list(self, node) -> Optional[List[float]]:
        lst = literal_list(node)
        if lst is None and hasattr(node, "id") and node.id in self.idx.literal_lists:
            lst = self.idx.literal_lists[node.id][1]
        return lst

    def _images_without_code(self) -> List[RiskFinding]:
        out = []
        for cell in self.nb.code_cells:
            if cell.images and cell.cell_index not in self.plot_cells and not re.search(r"plot|show|display|Image|savefig|imshow", cell.source_code):
                out.append(_f("image_without_code", "info", f"Cell {cell.cell_index} shows an image that no plotting code produced", [cell.cell_index],
                              "The figure may be pasted or generated elsewhere; its provenance cannot be traced."))
        return out

    # ------------------------------------------------------------------ claims vs plots
    def matching_series(self, claim: Claim) -> List[SeriesEvidence]:
        ckinds = {k for n in claim.numbers for k in n.kinds} | kinds_of_text(claim.text)
        ctoks = content_tokens(claim.text)
        hits = []
        for s in self.series:
            skinds = kinds_of_text(s.label)
            if (ckinds and skinds & ckinds) or len(ctoks & content_tokens(s.label)) >= 2:
                hits.append(s)
        return hits

    def _trend_consistency(self) -> List[RiskFinding]:
        out = []
        for claim in self.claims:
            if not claim.direction or "comparison" in claim.tags:
                continue
            for s in self.matching_series(claim):
                if s.direction in ("increase", "decrease", "stable") and s.direction != claim.direction:
                    out.append(_f("plot_trend_mismatch", "high",
                                  f"{claim.claim_id} says '{claim.direction}' but the data plotted in Cell {s.cell_index} ({s.label}) shows '{s.direction}'",
                                  [s.cell_index], f"Plotted values: {s.values}", [claim.claim_id]))
                elif "monotonic" in claim.tags and s.direction == claim.direction and not s.monotonic:
                    out.append(_f("plot_trend_mismatch", "medium",
                                  f"{claim.claim_id} claims a steady/consistent {claim.direction}, but the plotted series in Cell {s.cell_index} is not monotonic",
                                  [s.cell_index], f"Plotted values: {s.values}", [claim.claim_id]))
        return out

    def _dimred_claims(self) -> List[RiskFinding]:
        out = []
        dimred = [c for c in self.idx.calls if DIMRED_RE.fullmatch(c.attr or "")]
        if not dimred:
            return out
        cells = sorted({c.cell_index for c in dimred})
        for c in dimred:
            if c.attr == "TSNE" and "random_state" not in c.kwargs:
                out.append(_f("tsne_unseeded", "low", f"t-SNE in Cell {c.cell_index} has no random_state; the layout changes run to run", [c.cell_index], f"`{c.source_line}`"))
        related = [cl.claim_id for cl in self.claims if re.search(r"t-?sne|umap|cluster|separat|projection|embedding space", cl.text, re.I)]
        if related:
            out.append(_f("projection_interpretation", "medium",
                          "Claims interpret a 2-D projection (t-SNE/UMAP/PCA) as evidence of class structure",
                          cells, "Distances and cluster sizes in t-SNE/UMAP are not faithful; back separation claims with a quantitative metric (silhouette, kNN accuracy, linear probe).",
                          related))
        return out

    def figure_claims(self) -> List[str]:
        return [c.claim_id for c in self.claims if FIGURE_REF_RE.search(c.text)]
