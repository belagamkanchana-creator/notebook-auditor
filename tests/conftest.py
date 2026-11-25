import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXAMPLES = ROOT / "examples"


def _load_builder():
    spec = importlib.util.spec_from_file_location("make_sample_notebook", EXAMPLES / "make_sample_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def sample_nb_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("nb") / "sample.ipynb"
    path.write_text(json.dumps(_load_builder().nb), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="session")
def slides_path():
    return str(EXAMPLES / "slides.md")


def make_nb(cells, tmp_path, name="nb.ipynb"):
    """Build a minimal notebook file from (cell_type, source, outputs) tuples."""
    raw = []
    for i, (ctype, src, outputs) in enumerate(cells, start=1):
        cell = {"cell_type": ctype, "metadata": {}, "source": src}
        if ctype == "code":
            cell["execution_count"] = i
            cell["outputs"] = [
                {"output_type": "stream", "name": "stdout", "text": o} if isinstance(o, str) else o for o in outputs
            ]
        raw.append(cell)
    path = Path(tmp_path) / name
    path.write_text(json.dumps({"cells": raw, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    return str(path)
