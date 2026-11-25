"""Static (AST) index of the notebook's code cells: calls, assignments and literals."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .models import Notebook

_MAGIC_RE = re.compile(r"^\s*([%!]|\?\w|\w+\?\s*$)")


def clean_source(src: str) -> str:
    """Replace IPython magics / shell escapes with `pass` so the cell parses (line numbers preserved)."""
    out = []
    for line in src.splitlines():
        if _MAGIC_RE.match(line):
            indent = len(line) - len(line.lstrip())
            out.append(" " * indent + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return f"{dotted(node.func)}()"
    if isinstance(node, ast.Subscript):
        return f"{dotted(node.value)}[]"
    return type(node).__name__


def names_in(node: Optional[ast.AST]) -> Set[str]:
    if node is None:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def target_names(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: List[str] = []
        for elt in node.elts:
            out.extend(target_names(elt))
        return out
    if isinstance(node, ast.Starred):
        return target_names(node.value)
    return []


def literal_number(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = literal_number(node.operand)
        return -v if v is not None else None
    return None


def literal_list(node: ast.AST) -> Optional[List[float]]:
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = [literal_number(e) for e in node.elts]
        if vals and all(v is not None for v in vals):
            return [float(v) for v in vals]  # type: ignore[arg-type]
    if isinstance(node, ast.Call) and dotted(node.func).split(".")[-1] in ("array", "asarray", "Series", "tensor") and node.args:
        return literal_list(node.args[0])
    return None


@dataclass
class CallEvent:
    cell_index: int
    lineno: int
    col: int
    func: str  # dotted call name, e.g. "scaler.fit_transform"
    attr: str  # last component, e.g. "fit_transform"
    receiver: Optional[str]  # dotted receiver, e.g. "scaler"
    receiver_call: Optional[str]  # receiver's constructor if receiver is itself a call: StandardScaler().fit(...)
    args: List[str]
    arg_names: List[Set[str]]
    kwargs: Dict[str, str]
    kwarg_values: Dict[str, Optional[float]]
    targets: List[str]
    source_line: str
    arg_nodes: List[ast.AST] = field(default_factory=list, repr=False, compare=False)

    @property
    def order(self) -> Tuple[int, int, int]:
        return (self.cell_index, self.lineno, self.col)

    @property
    def all_arg_names(self) -> Set[str]:
        out: Set[str] = set()
        for s in self.arg_names:
            out |= s
        return out


@dataclass
class AssignEvent:
    cell_index: int
    lineno: int
    targets: List[str]
    refs: Set[str]
    value_src: str

    @property
    def order(self) -> Tuple[int, int, int]:
        return (self.cell_index, self.lineno, 0)


class CodeIndex:
    def __init__(self, nb: Notebook) -> None:
        self.nb = nb
        self.calls: List[CallEvent] = []
        self.assigns: List[AssignEvent] = []
        self.var_class: Dict[str, Tuple[str, Tuple[int, int, int]]] = {}
        self.literal_lists: Dict[str, Tuple[int, List[float]]] = {}
        self.numeric_assigns: List[Tuple[int, int, str, float]] = []  # (cell, line, name, value)
        self.compares: List[Tuple[int, int, str, str, float]] = []  # (cell, line, left_src, op, constant)
        self.parse_errors: List[int] = []
        self.sources: Dict[int, str] = {}
        for cell in nb.code_cells:
            self.sources[cell.cell_index] = cell.source_code
            self._index_cell(cell.cell_index, cell.source_code)
        self.calls.sort(key=lambda c: c.order)
        self.assigns.sort(key=lambda a: a.order)

    # ------------------------------------------------------------------ building
    def _index_cell(self, idx: int, src: str) -> None:
        cleaned = clean_source(src)
        try:
            tree = ast.parse(cleaned)
        except SyntaxError:
            self.parse_errors.append(idx)
            return
        lines = cleaned.splitlines()
        assign_targets: Dict[int, List[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
                names: List[str] = []
                for t in tgts:
                    names.extend(target_names(t))
                value = node.value
                if value is None:
                    continue
                if isinstance(value, ast.Call):
                    assign_targets[id(value)] = names
                    ctor = dotted(value.func).split(".")[-1]
                    for n in names:
                        self.var_class[n] = (ctor, (idx, node.lineno, 0))
                lst = literal_list(value)
                if lst is not None:
                    for n in names:
                        self.literal_lists[n] = (idx, lst)
                num = literal_number(value)
                if num is not None:
                    for n in names:
                        self.numeric_assigns.append((idx, node.lineno, n, num))
                self.assigns.append(
                    AssignEvent(idx, node.lineno, names, names_in(value), _segment(lines, value))
                )
            elif isinstance(node, ast.Compare):
                for op, comp in zip(node.ops, node.comparators):
                    left, right = node.left, comp
                    for a, b in ((left, right), (right, left)):
                        v = literal_number(b)
                        if v is not None and literal_number(a) is None:
                            self.compares.append((idx, node.lineno, _segment(lines, a), type(op).__name__, v))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = dotted(node.func)
            attr = func.split(".")[-1].rstrip("()")
            receiver = receiver_call = None
            if isinstance(node.func, ast.Attribute):
                receiver = dotted(node.func.value)
                if isinstance(node.func.value, ast.Call):
                    receiver_call = dotted(node.func.value.func).split(".")[-1]
            kwargs = {k.arg: _segment(lines, k.value) for k in node.keywords if k.arg}
            kwarg_values = {k.arg: literal_number(k.value) for k in node.keywords if k.arg}
            self.calls.append(
                CallEvent(
                    cell_index=idx,
                    lineno=node.lineno,
                    col=node.col_offset,
                    func=func,
                    attr=attr,
                    receiver=receiver,
                    receiver_call=receiver_call,
                    args=[_segment(lines, a) for a in node.args],
                    arg_names=[names_in(a) for a in node.args],
                    kwargs=kwargs,
                    kwarg_values=kwarg_values,
                    targets=assign_targets.get(id(node), []),
                    source_line=lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else "",
                    arg_nodes=list(node.args),
                )
            )

    # ------------------------------------------------------------------ queries
    def calls_to(self, *names: str) -> List[CallEvent]:
        wanted = set(names)
        return [c for c in self.calls if c.attr in wanted]

    def class_of(self, call: CallEvent) -> Optional[str]:
        """Constructor class of the object a method is called on, if known."""
        if call.receiver_call:
            return call.receiver_call
        if call.receiver and call.receiver in self.var_class:
            ctor, pos = self.var_class[call.receiver]
            return ctor
        return None

    def taint_forward(self, names: Set[str], after: Tuple[int, int, int]) -> Set[str]:
        """Propagate a set of variable names through later assignments (one-pass data flow)."""
        tainted = set(names)
        for a in self.assigns:
            if a.order <= after:
                continue
            if a.refs & tainted:
                tainted |= set(a.targets)
        return tainted

    def source_matches(self, pattern: str, flags: int = re.I) -> List[Tuple[int, int, str]]:
        rx = re.compile(pattern, flags)
        hits = []
        for idx, src in self.sources.items():
            for ln, line in enumerate(src.splitlines(), start=1):
                if rx.search(line):
                    hits.append((idx, ln, line.strip()))
        return hits


def _segment(lines: List[str], node: ast.AST) -> str:
    try:
        return ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - py<3.9 fallback
        ln = getattr(node, "lineno", 1)
        return lines[ln - 1].strip() if 0 < ln <= len(lines) else ""
