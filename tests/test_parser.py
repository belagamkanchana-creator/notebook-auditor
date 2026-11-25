from notebook_auditor.parser import load_notebook, parse_notebook_dict

from conftest import make_nb


def test_parses_cells_outputs_and_images(sample_nb_path):
    nb = load_notebook(sample_nb_path)
    assert nb.cells[0].cell_type == "markdown"
    load = nb.get(3)
    assert load.cell_type == "code" and load.execution_count == 2
    assert "Loaded 420 documents" in load.output_text
    assert [o.output_type for o in load.outputs] == ["stream", "execute_result"]
    plot = nb.get(9)
    assert plot.images and plot.images[0].mime == "image/png"
    assert (plot.images[0].width, plot.images[0].height) == (64, 48)


def test_linear_text_has_cell_ids(sample_nb_path):
    text = load_notebook(sample_nb_path).to_linear_text()
    assert "[Cell 3 - Code] (In [2])" in text
    assert "[Cell 3 - Output]" in text
    assert "[Cell 1 - Markdown]" in text
    assert "<image/png 64x48" in text


def test_html_tables_errors_and_list_sources():
    nb = parse_notebook_dict(
        {
            "nbformat": 4,
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "source": ["df.describe()\n", ""],
                    "outputs": [
                        {
                            "output_type": "execute_result",
                            "data": {
                                "text/plain": "...",
                                "text/html": "<table><tr><th></th><th>acc</th></tr><tr><th>mean</th><td>0.91</td></tr></table>",
                            },
                        },
                        {"output_type": "error", "ename": "KeyError", "evalue": "'x'", "traceback": ["\x1b[31mKeyError\x1b[0m"]},
                    ],
                }
            ],
        }
    )
    cell = nb.cells[0]
    assert cell.source_code == "df.describe()\n"
    assert cell.outputs[0].html_tables == [[["", "acc"], ["mean", "0.91"]]]
    assert cell.has_error and cell.outputs[1].traceback == "KeyError"


def test_progress_bars_collapsed(tmp_path):
    path = make_nb([("code", "train()", ["epoch 1\r 50%\r100% done\nfinal loss: 0.3\n"])], tmp_path)
    out = load_notebook(path).cells[0].output_text
    assert "100% done" in out and "50%" not in out
