# notebook-auditor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/belagamkanchana-creator/notebook-auditor/blob/main/colab/Notebook_Auditor.ipynb)

**No install:** click the badge, run the three cells, and audit the built-in example or your own notebook (upload it or pick it from Google Drive).

Audits a Jupyter/Colab notebook (`.ipynb`) and the slides, paper or summary written from it. It checks three things:

- **Traceability:** whether every claim traces to a cell output.
- **Discrepancies:** whether the numbers and trends in the text match what the cells print.
- **Methodology:** whether there are risks the outputs hide (leakage, missing baselines, borderline thresholds, misleading figures).

```bash
pip install -e .                    # no required dependencies
pip install -e '.[anthropic]'       # optional: LLM agents (also: [openai], [litellm])

notebook-auditor run --notebook path/to/nb.ipynb --slides path/to/slides.md --output-dir ./audit_reports
```

This writes `audit_report.md` (a readable report with links to the evidence cells) and `audit_report.json` (claims, plans, evidence traces, verdicts and findings).

Try it on the bundled synthetic example, which has planted problems:

```bash
python examples/make_sample_notebook.py
notebook-auditor run -n examples/sample_notebook.ipynb -s examples/slides.md -o ./audit_reports --provider none
```

## How it works

```
parser ──► claim extractor ──► Planner ──► Executor ──► Verifier ──► auditors ──► report
 .ipynb      slides / md        claim→cells   exact values   strict rules   evidence · risk · strength · figure
```

| Module | What it does |
|---|---|
| `parser.py` | Parses cells, `stream`/`execute_result`/`display_data`/`error` outputs, HTML tables and PNG/JPEG/SVG metadata. Produces a linear view with `[Cell N - Code]` / `[Cell N - Output]` markers (`notebook-auditor parse -n nb.ipynb`). |
| `claim_extractor.py` | Turns slides and markdown into atomic claims: splits compound sentences, resolves pronouns ("It reaches…" becomes "Our model reaches…"), turns table rows into claims, and skips structural headings. Each claim gets a category: `numeric`, `trend`, `methodological` or `qualitative`. |
| `pipeline.py` | **Planner** maps each claim to its target cells. **Executor** extracts the printed values, shapes, series and code parameters. **Verifier** applies strict rules and assigns `SUPPORTED`, `CONTRADICTED`, `UNVERIFIABLE` or `METHODOLOGICAL_RISK`. |
| `rules.py` | The deterministic verification rules (see below). |
| `auditors/evidence_auditor.py` | Labels every printed number with what it measures (from text labels, `classification_report` and `to_string()` tables, and DataFrame HTML). Checks claims for match, derived, mismatch or untraced. |
| `auditors/risk_auditor.py` | Checks for leakage (transformers fitted or resampled before the split, test-set fitting, group-ID leakage, no hold-out), missing or near-chance baselines, p-values or metrics within ±0.005 of α or a threshold, execution order, errors and seeds. |
| `auditors/claim_strength_auditor.py` | Rates each claim `MEASURED`, `DERIVED` or `INTERPRETATION`, adds a "What this does not prove" limit, and flags overclaiming. |
| `auditors/figure_auditor.py` | Checks axis labels, truncated or out-of-range axes, unrendered figures, twin axes and unseeded t-SNE. Also checks that the trend in literally plotted data matches the claim. |

### Verification rules (abridged)

- **N-MATCH / N-MISMATCH:** a number is supported only if a cell prints it under a compatible label: same metric (accuracy ≠ F1), same split (train ≠ test), rounding of the last stated digit allowed. A mismatch needs that same compatible label. Broad labels like counts also need a shared word, so "252 positive reviews" is not compared with "Loaded 420 documents".
- **N-DERIVED:** the number equals a difference, ratio or relative change of two same-kind outputs.
- **T-DIRECTION / T-MONOTONIC:** the trend must match printed or plotted series. "Steadily" requires a monotonic series, with bumps under 2% of the range tolerated.
- **M-\*:** procedure claims are checked against the code: stratify, test fraction (including sklearn's 0.25 default), fold count, seeds, baseline, scaling, held-out split.
- **RISK-OVERLAY:** a matching number becomes `METHODOLOGICAL_RISK` when high-severity leakage, near-chance accuracy, or a borderline threshold undermines the conclusion drawn from it.

### LLM mode and its guardrails

With `--provider auto` (the default), the tool uses `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` if one is set. Otherwise it runs fully offline. `--provider litellm --model <any>` works too. Default models are `claude-sonnet-5-5` and `gpt-4o-mini`; override them with `--model`.

The LLM refines claim extraction, planning and extraction, and settles `UNVERIFIABLE` cases. Its output is always re-checked:

- A rewritten claim containing numbers that aren't in the source is dropped.
- Planned cell indices outside the notebook are discarded.
- Extracted values must be **verbatim quotes** from the cell, or they are rejected.
- An LLM verdict must cite a quote that exists in a target cell. It can never "support" a number that no cited quote contains, and it cannot override the deterministic numeric rules.

API failures fall back to the heuristics and are listed under **Warnings** in the report.

### CLI options

```
notebook-auditor run  -n NB [-s ARTIFACT ...] [-o DIR] [--provider auto|anthropic|openai|litellm|none]
                      [--model M] [--include-notebook-markdown] [--threshold-tolerance 0.005]
                      [--format md|json|both] [--fail-on none|contradicted|risk|any]
notebook-auditor parse  -n NB          # linear text view
notebook-auditor claims -s ARTIFACT    # extracted claims as JSON
```

With no `--slides`, claims are taken from the notebook's own markdown cells. `--fail-on` makes the exit code non-zero, which is useful in CI.

**Cell numbering:** `Cell N` is the 1-based position in the notebook, markdown cells included.

## Limits

- Static analysis only. The notebook is never executed, so verdicts reflect the *saved* outputs.
- Images are checked through their plotting code and metadata, not by looking at pixels.
- Leakage detection follows simple data flow through assignments. Aliasing through functions or classes can hide it.
- The heuristic claim extractor is rule-based. Use an LLM provider for messy prose.

## Development

```bash
pip install -e '.[dev]' && pytest
```
