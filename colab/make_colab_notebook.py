"""Build colab/Notebook_Auditor.ipynb — the notebook behind the "Open in Colab" link."""
import json
from pathlib import Path

REPO = "https://github.com/belagamkanchana-creator/notebook-auditor.git"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text, title):
    src = f"#@title {title}\n" + text.strip("\n")
    return {
        "cell_type": "code",
        "metadata": {"cellView": "form"},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


INTRO = """
# Notebook Auditor

Upload a research notebook and get back:

- **What your notebook found:** dataset size, class balance, final scores, trends across epochs or settings, overfitting, which model or class does best and worst. Each finding links to its cell and gets a trust rating: ✅ solid, ⚠️ at risk, ❔ unconfirmed or 👁️ read from a figure.
- **Why a result might not hold:** data leakage, missing baselines, p-values right at 0.05, typed-in chart values, misleading axes.
- **What's missing:** concrete next steps with code, such as adding a baseline, repeating with several seeds, or reporting a confidence interval.
- **A check of your write-up (optional):** add your slides or paper draft and every claim is traced to the cell that proves or disproves it.

**How to use it:** run the three cells below in order (click ▶ on each). Start with the built-in example to see what a report looks like.

With Claude switched off, everything runs inside this Colab session and your notebook is not sent anywhere. Switch Claude on (step 3) to also read your graphs and draft richer findings.
"""

SETUP = f"""
import subprocess, sys, os
if not os.path.isdir("notebook-auditor"):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "{REPO}"], check=True)
else:
    subprocess.run(["git", "-C", "notebook-auditor", "pull", "-q"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "./notebook-auditor[anthropic]"], check=True)
print("Ready. Now run step 2.")
"""

CHOOSE = """
source = "Try the built-in example"  #@param ["Try the built-in example", "Upload from my computer", "From Google Drive"]
#@markdown **From Google Drive only:** paths to your notebook and (optional) write-up. Save the notebook (Ctrl+S) first so its outputs are included.
drive_notebook = "/content/drive/MyDrive/Colab Notebooks/MyNotebook.ipynb"  #@param {type:"string"}
drive_writeup = ""  #@param {type:"string"}

nb_path, writeups = None, []
if source.startswith("Try"):
    nb_path = "notebook-auditor/examples/sample_notebook.ipynb"
    writeups = ["notebook-auditor/examples/slides.md"]
elif source.startswith("Upload"):
    from google.colab import files
    print("Pick your .ipynb file. You can also pick your slides or paper as .md or .txt (hold Ctrl or Cmd to select several).")
    for name in files.upload():
        if name.endswith(".ipynb"):
            nb_path = name
        else:
            writeups.append(name)
else:
    from google.colab import drive
    drive.mount("/content/drive")
    nb_path = drive_notebook
    writeups = [drive_writeup] if drive_writeup.strip() else []

if not nb_path:
    raise SystemExit("No .ipynb file was selected. Run this cell again and pick a notebook.")
print("Notebook:", nb_path)
print("Write-up:", ", ".join(writeups) if writeups else "none (fine: the report lists what the notebook found)")
"""

RUN = """
use_claude = False  #@param {type:"boolean"}
#@markdown With Claude on, it also reads your graphs and drafts richer findings; every number it reports is still checked against the outputs. It needs an `ANTHROPIC_API_KEY` saved under the 🔑 **Secrets** tab on the left. Your notebook (code, outputs and figures) is sent to the Anthropic API when this is on.

import os
from IPython.display import Markdown, display
from notebook_auditor import run_audit
from notebook_auditor.parser import load_notebook
from notebook_auditor.report import write_reports

llm = None
if use_claude:
    try:
        from google.colab import userdata
        os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
        from notebook_auditor.llm import make_client
        llm = make_client("anthropic")
    except Exception as exc:
        print(f"Claude is not available ({type(exc).__name__}). Add ANTHROPIC_API_KEY under Secrets. Running the offline checks instead.\\n")

result = run_audit(nb_path, writeups, provider="none", llm=llm, findings=True)
paths = write_reports(result, "audit_report", load_notebook(nb_path))
s = result.summary()
t, v = s["findings_by_trust"], s["verdicts"]
print(f"Findings: {s['findings']} (✅ {t['solid']} solid · ⚠️ {t['at_risk']} at risk · ❔ {t['unconfirmed']} unconfirmed · 👁️ {t['visual']} from figures) · next steps: {s['gaps']}")
if writeups:
    print(f"Write-up claims: {len(result.claims)} (✅ {v['SUPPORTED']} supported · ❌ {v['CONTRADICTED']} contradicted · "
          f"⚠️ {v['METHODOLOGICAL_RISK']} at risk · ❔ {v['UNVERIFIABLE']} unverifiable)")
print("Mode:", result.mode)
print("Report files: audit_report/audit_report.md and audit_report/audit_report.json (Files tab on the left)\\n")
display(Markdown(open(paths["md"], encoding="utf-8").read()))
"""

DOWNLOAD = """
from google.colab import files
files.download("audit_report/audit_report.md")
files.download("audit_report/audit_report.json")
"""

nb = {
    "cells": [
        md(INTRO),
        code(SETUP, "1 · Set up (run once, about 20 seconds)"),
        code(CHOOSE, "2 · Choose what to audit"),
        code(RUN, "3 · Run the audit"),
        md("### Optional: download the report"),
        code(DOWNLOAD, "Download audit_report.md and audit_report.json"),
    ],
    "metadata": {
        "colab": {"provenance": [], "name": "Notebook_Auditor.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

if __name__ == "__main__":
    out = Path(__file__).with_name("Notebook_Auditor.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
