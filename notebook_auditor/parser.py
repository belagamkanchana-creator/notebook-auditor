"""Load `.ipynb` files into structured :class:`Notebook` objects."""
from __future__ import annotations

import base64
import json
import re
import struct
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .models import Cell, CellOutput, ImageInfo, Notebook

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _join(src: Any) -> str:
    if src is None:
        return ""
    if isinstance(src, list):
        return "".join(str(s) for s in src)
    return str(src)


def clean_stream_text(text: str) -> str:
    """Strip ANSI colour codes and collapse carriage-return progress bars."""
    text = ANSI_RE.sub("", text)
    lines = []
    for line in text.split("\n"):
        if "\r" in line:
            segments = [s for s in line.split("\r") if s.strip()]
            line = segments[-1] if segments else ""
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- HTML helpers
class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[List[List[str]]] = []
        self._depth = 0
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
            self.tables.append([])
        elif tag == "tr" and self._depth:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row and self.tables:
                self.tables[-1].append(self._row)
            self._row = None
        elif tag == "table":
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


class _TextStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self._skip += 1
        elif tag in ("br", "p", "tr", "div", "li"):
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def parse_html_tables(html: str) -> List[List[List[str]]]:
    parser = _TableParser()
    try:
        parser.feed(html)
    except Exception:  # malformed HTML should never kill the audit
        return []
    return [t for t in parser.tables if t]


def html_to_text(html: str) -> str:
    parser = _TextStripper()
    try:
        parser.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    text = "".join(parser.parts)
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


# --------------------------------------------------------------------------- images
def _jpeg_size(raw: bytes) -> Tuple[Optional[int], Optional[int]]:
    i = 2
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while i < len(raw) - 9:
        if raw[i] != 0xFF:
            i += 1
            continue
        marker = raw[i + 1]
        if marker in sof:
            h, w = struct.unpack(">HH", raw[i + 5 : i + 9])
            return w, h
        if marker in (0xD8, 0x01, 0xFF) or 0xD0 <= marker <= 0xD7:
            i += 2 if marker != 0xFF else 1
            continue
        seg_len = struct.unpack(">H", raw[i + 2 : i + 4])[0]
        i += 2 + seg_len
    return None, None


def image_info(mime: str, payload: Any) -> ImageInfo:
    data = _join(payload)
    if mime == "image/svg+xml":
        m = re.search(r'width="([\d.]+)(?:pt|px)?"[^>]*height="([\d.]+)', data)
        w, h = (int(float(m.group(1))), int(float(m.group(2)))) if m else (None, None)
        return ImageInfo(mime, len(data.encode()), w, h)
    try:
        raw = base64.b64decode(data)
    except Exception:
        return ImageInfo(mime, 0)
    w = h = None
    try:
        if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
            w, h = struct.unpack(">II", raw[16:24])
        elif raw[:2] == b"\xff\xd8":
            w, h = _jpeg_size(raw)
    except struct.error:
        pass
    return ImageInfo(mime, len(raw), w, h)


# --------------------------------------------------------------------------- outputs & cells
_IMAGE_KEYS = {
    "image/png": "image/png",
    "png": "image/png",
    "image/jpeg": "image/jpeg",
    "jpeg": "image/jpeg",
    "image/svg+xml": "image/svg+xml",
    "svg": "image/svg+xml",
}


def parse_output(out: Dict[str, Any]) -> CellOutput:
    otype = out.get("output_type", "")
    if otype == "stream":
        return CellOutput("stream", text=clean_stream_text(_join(out.get("text"))), stream_name=out.get("name", "stdout"))

    if otype in ("execute_result", "display_data", "pyout"):
        data = out.get("data")
        if data is None:  # nbformat 3 stores mime bundles at the top level
            data = {k: v for k, v in out.items() if k not in ("output_type", "metadata", "prompt_number")}
        text_plain = _join(data.get("text/plain", data.get("text")))
        html = _join(data.get("text/html", data.get("html")))
        tables = parse_html_tables(html) if html and "<table" in html.lower() else []
        images = [image_info(mime, data[key]) for key, mime in _IMAGE_KEYS.items() if key in data]
        text = text_plain
        if not text and html:
            text = html_to_text(html)
        return CellOutput(
            "execute_result" if otype == "pyout" else otype,
            text=clean_stream_text(text),
            html_tables=tables,
            images=images,
        )

    if otype in ("error", "pyerr"):
        ename = out.get("ename", "Error")
        evalue = out.get("evalue", "")
        tb = ANSI_RE.sub("", "\n".join(out.get("traceback") or []))
        return CellOutput("error", text=f"{ename}: {evalue}", error_name=ename, traceback=tb)

    return CellOutput(otype or "unknown", text=_join(out.get("text")))


def _parse_cell(raw: Dict[str, Any], index: int) -> Cell:
    ctype = raw.get("cell_type", "code")
    if ctype == "heading":  # nbformat 3
        ctype = "markdown"
    source = _join(raw.get("source", raw.get("input")))
    outputs = [parse_output(o) for o in raw.get("outputs", []) or []] if ctype == "code" else []
    exec_count = raw.get("execution_count", raw.get("prompt_number"))
    return Cell(cell_index=index, cell_type=ctype, source_code=source, execution_count=exec_count, outputs=outputs)


def parse_notebook_dict(nb: Dict[str, Any], path: str = "<memory>") -> Notebook:
    major = int(nb.get("nbformat", 4))
    if major < 4 and "worksheets" in nb:
        raw_cells = [c for ws in nb.get("worksheets", []) for c in ws.get("cells", [])]
    else:
        raw_cells = nb.get("cells", [])
    meta = nb.get("metadata", {}) or {}
    language = (
        (meta.get("kernelspec") or {}).get("language")
        or (meta.get("language_info") or {}).get("name")
        or "python"
    )
    cells = [_parse_cell(c, i) for i, c in enumerate(raw_cells, start=1)]
    return Notebook(path=path, cells=cells, language=language, nbformat=major)


def load_notebook(path: Union[str, Path]) -> Notebook:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid notebook JSON: {exc}") from exc
    if not isinstance(data, dict) or ("cells" not in data and "worksheets" not in data):
        raise ValueError(f"{path} does not look like a Jupyter notebook (no 'cells').")
    return parse_notebook_dict(data, str(path))


def notebook_summary(nb: Notebook) -> Dict[str, Any]:
    code = nb.code_cells
    counts = [c.execution_count for c in code if c.execution_count is not None]
    return {
        "cells": len(nb.cells),
        "code_cells": len(code),
        "markdown_cells": len(nb.markdown_cells),
        "executed_code_cells": len(counts),
        "cells_with_errors": [c.cell_index for c in code if c.has_error],
        "images": sum(len(c.images) for c in code),
        "language": nb.language,
    }
